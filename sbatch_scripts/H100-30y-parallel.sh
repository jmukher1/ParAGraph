#!/bin/bash
#SBATCH --time=05:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="1M-gpu-30y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1

export OMP_NUM_THREADS=16
cat /proc/cpuinfo
sh drivers/parallel_gpu_30y_static.sh
