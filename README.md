# Developing an Artificial Intelligence-Based Representation of a Virtual Patient Model for Real-Time Diagnosis of Acute Respiratory Distress Syndrome
This repository includes the code and data required to replicate the results of the manuscript "Developing an Artificial Intelligence-Based Representation of a Virtual Patient Model for Real-Time Diagnosis of Acute Respiratory Distress Syndrome."

Please note that some of the code is streamlined for parallel execution and will need to be adapted for serial implementation.

# Dataset
* The generated patient data is available for download from the [FZ-Juelich B2Share website](https://doi.org/10.23728/b2share.b143c287bb69482a90ababe7a5a8eb4a).

# Software, Packages, and versions
## Required for parallel execution
* cuDNN==8.0.2.39
* CUDA==11.0
* mpi4py==3.1.4

## Required for serial execution
* Python==3.8.5
* tensorflow==2.4.0
* matplotlib==3.4.1
* plotly==4.14.3
* numpy==1.19.2
* pandas==1.2.4
* ray==2.0.0
* scikit-learn==0.24.1
* scipy==1.7.3
* simplejson==3.17.2

# Repository Structure
* The [best_trials](best_trials/) directory contains the best parameter outputs from the hyperparameter tuning runs.
* The [saved_model](saved_model/) directory contains the models trained on the best parameters selected by each of the 4 schedulers.
* [training_results](training_results/) contains the training performance of the ANN and CNN models described in the paper as well as the 4 models contained in the [saved_model](saved_model/) directory. The contents of this directory are generated from [train_and_test_NN.ipynb](train_and_test_NN.ipynb).
* [figures](figures/) is the directory where figures from [visualize_results_plotly.ipynb](visualize_results_plotly.ipynb) are expected to be saved.
* Further directories can or will need to be created when [Ray Tune is initialised](ray_tune_experiment.py).
