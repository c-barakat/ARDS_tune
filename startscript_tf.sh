#!/bin/bash
#SBATCH --job-name=ards_tune_fio_mae
#SBATCH --output=ards_tune_fio_mae.out
#SBATCH --error=ards_tune_fio_mae.err

#SBATCH --nodes=16
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --exclusive


module --force purge
module use $OTHERSTAGES
module load Stages/Devel-2020
module load GCCcore/.9.3.0
module load GCC/9.3.0
module load Python/3.8.5
module load ParaStationMPI

module load CUDA/11.0
module load cuDNN/8.0.2.39-CUDA-11.0


source jupyter/kernels/gpu_kernel/bin/activate

sleep 1

export CUDA_VISIBLE_DEVICES="0"

# num_gpus=1
####### this part is taken from the ray example slurm script #####
set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# __doc_head_ray_start__
## port AHB 
# port=8182
## port HPB
# port=8962
## port PBT
# port=9640
## port FIO
port=8705

ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

export redis_total_address=$ip_head

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address $ip_head \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

echo "Ready"

# python3 -u  ray_tune_experiment.py --num_samples 64 --max_num_epochs 40 --gpus_per_trial 1 --cpus_per_trial 16 --scheduler ahb
# python3 -u  ray_tune_experiment.py --num_samples 64 --max_num_epochs 40 --gpus_per_trial 1 --cpus_per_trial 16 --scheduler hpb
# python3 -u  ray_tune_experiment.py --num_samples 64 --max_num_epochs 40 --gpus_per_trial 1 --cpus_per_trial 16 --scheduler pbt
python3 -u  ray_tune_experiment.py --num_samples 64 --max_num_epochs 40 --gpus_per_trial 1 --cpus_per_trial 16
