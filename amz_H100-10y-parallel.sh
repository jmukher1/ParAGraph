#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="amz-pr-10y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1 

source ~/.bashrc

nvidia-smi

export OMP_NUM_THREADS=16
sh drivers/amz_parallel_gpu_H100_10y_static.sh


