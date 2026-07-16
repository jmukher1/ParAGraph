#!/bin/bash
#SBATCH --time=05:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="gpu-mrun-10y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1 

source ~/.bashrc

export OMP_NUM_THREADS=16
cat /proc/cpuinfo
sh drivers/mass_cuda_multi_run.sh