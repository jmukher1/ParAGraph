#!/bin/bash
#SBATCH --time=08:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name="cpu-pr30y"
#SBATCH --account=ayg
#SBATCH --partition=smallgpu
#SBATCH --gres=gpu:l40:1 


source ~/.bashrc

cat /proc/cpuinfo

export OMP_NUM_THREADS=16
sh drivers/parallel_30y_profile_static.sh
