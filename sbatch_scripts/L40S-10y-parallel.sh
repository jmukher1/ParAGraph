#!/bin/bash
#SBATCH --time=05:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name="gpu-pr-10y"
#SBATCH --account=ayg
#SBATCH --partition=smallgpu
#SBATCH --gres=gpu:l40:1 

source ~/.bashrc
nvidia-smi

export OMP_NUM_THREADS=16
sh drivers/parallel_gpu_l40_10y_static.sh
