#!/bin/bash
#SBATCH --time=08:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="abl-pr-10y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1 

source ~/.bashrc
export OMP_NUM_THREADS=16
cat /proc/cpuinfo
sh drivers/ablation_parallel_gpu_10y.sh
