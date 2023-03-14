import glob
import os
import simplejson

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import ray
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandScheduler,
    PopulationBasedTraining,
)
# from tensorflow import keras
from tensorflow.keras import Input, layers, models
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D, MaxPooling1D
# from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam

import argparse


def get_model(input_size, dropout_rate=0.5, final_layer=False):
    inputs = Input(shape=(input_size, 1), name="Input_Layer")

    layer = layers.Conv1D(64, 9, activation="relu", name="Conv_Layer_1")(inputs)
    layer = layers.Conv1D(
        128,
        5,
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer="l2",
        name="Conv_Layer_2",
    )(layer)
    layer = layers.Dropout(dropout_rate, name="Dropout_Layer_1")(layer)
    layer = layers.Conv1D(
        128,
        5,
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer="l2",
        name="Conv_Layer_3",
    )(layer)
    layer = layers.Dropout(dropout_rate, name="Dropout_Layer_2")(layer)
    layer = layers.Conv1D(
        128,
        3,
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer="l2",
        name="Conv_Layer_4",
    )(layer)
    layer = layers.Flatten(name="Flatten_Layer")(layer)

    if final_layer:
        layer = layers.Dense(20, activation="relu", name="Fully_Connected_Layer")(layer)

    outputs = layers.Dense(4, name="Output_Layer")(layer)
    CNN_model = models.Model(inputs=inputs, outputs=outputs, name="CNN")

    return CNN_model


def get_scheduler(scheduler_name, num_epochs):
    if scheduler_name == 'pbt':
        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_mae",
#             metric="val_mse",
            mode="min",
            perturbation_interval=int(num_epochs/2),
            hyperparam_mutations={
                # "l1_scale": lambda: np.random.uniform(1e-3, 1e-5),
                "learning_rate": tune.qloguniform(1e-5, 1e-4, 1e-5)
            },
        )
    if scheduler_name == 'ahb':
        return AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="val_mae",
#             metric="val_mse",
            mode="min",
            grace_period=int(num_epochs/2),
            max_t=num_epochs,
        )
    if scheduler_name == 'hpb':
        return HyperBandScheduler(
            time_attr="training_iteration",
            metric="val_mae",
#             metric="val_mse",
            mode="min",
            max_t=num_epochs,
        )


def train_model(model, X_train, y_train, X_val, y_val, num_epochs, batch_size):
    callbacks = [TuneReportCallback()]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return history


def train_function(config, data, checkpoint_dir=None):
    model = get_model(
        data[0].shape[1],
        config["dropout_rate"],
        config["final_layer"],
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=config["loss_function"],
        metrics=["mse", "mae"],
    )

    history = train_model(
        model,
        data[0],
        data[1],
        data[2],
        data[3],
        config["num_epochs"],
        config["batch_size"],
    )

    tune.report(
        loss=history.history["loss"],
        val_mse=history.history["val_mse"],
        val_mae=history.history["val_mae"],
    )


def main(
    num_samples,
    max_num_epochs,
    gpus_per_trial,
    cpus_per_trial,
    scheduler,
):
    base_path = '/p/project/ai4hc/barakat1/med_data/sim_outputs'
    patient_data = pd.read_csv(f"{base_path}/patient_data.csv")

    # convert PaO2 and PaCO2 from kPa to mmHg
    patient_data.PaO2 = patient_data.PaO2 * 7.50062
    patient_data.PaCO2 = patient_data.PaCO2 * 7.50062

    def scale_data(patient_data):
        # Standardise the Features
        scaler = QuantileTransformer(
            n_quantiles=10000,
            output_distribution="normal",
        )

        patient_data_norm = pd.DataFrame(
            scaler.fit_transform(patient_data), columns=patient_data.columns
        )

        return patient_data_norm

    scaled_data = scale_data(patient_data)
# 
    def shape_data(scaled_data):
        # Split data into train and val sets
        train_data, val_data = train_test_split(
            scaled_data,
            test_size=0.1,
            shuffle=True,
        )

        # Split all datasets into input and expected outputs
        x_train, y_train = train_data.iloc[:, :-4].values, train_data.iloc[:, -4:].values
        x_val, y_val = val_data.iloc[:, :-4].values, val_data.iloc[:, -4:].values

        # this shaping is for ray tune
        return [x_train, y_train, x_val, y_val]

    all_data = shape_data(scaled_data)

    # connect to the ray cluster
    ray.init(address=os.environ['redis_total_address'])

    analysis = tune.run(
        tune.with_parameters(train_function, data=all_data),
        local_dir=os.path.join(
            os.path.abspath(os.getcwd()),
            "ray_results",
        ),
        resources_per_trial={"gpu": gpus_per_trial, "cpu": cpus_per_trial},
        num_samples=num_samples,
        config={
            "learning_rate": tune.qloguniform(1e-5, 1e-3, 1e-5),
            "loss_function": tune.choice(["mse", "mae"]),
            "num_epochs": max_num_epochs,
            "batch_size": tune.choice([64, 128, 256]),
            "dropout_rate": tune.quniform(0.5, 0.8, 0.02),
            "final_layer": tune.choice([True, False]),
        },
        scheduler=get_scheduler(scheduler, max_num_epochs),
        name=f"RayTune_ARDS_{scheduler}",
    )

    # find best results and write it to json
    best_trial = analysis.get_best_trial("val_mse", "min")
#     best_trial = analysis.get_best_trial("val_mae", "min")
    trial_id = best_trial.last_result['trial_id']
    best_trial_path = f"{base_path}/best_trials/best_trial_{trial_id}_{scheduler}.json"
    with open(best_trial_path, "w") as f:
        simplejson.dump(best_trial.last_result, f, ignore_nan=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        help="Number of trials to run.",
    )
    parser.add_argument(
        "--max_num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--gpus_per_trial",
        default=1,
        type=int,
        help="Number of GPUs to dedicate for each trial.",
    )
    parser.add_argument(
        "--cpus_per_trial",
        default=1,
        type=int,
        help="Number of CPUs to dedicate for each trial.",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        help="Scheduler to be used during tuning.",
    )

    args = parser.parse_args()
    # You can change the number of GPUs per trial here:
    main(
        num_samples=args.num_samples,
        max_num_epochs=args.max_num_epochs,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        scheduler=args.scheduler,
    )
