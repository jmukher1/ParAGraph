#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-pro-10y-parallel-pa"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=16
sh drivers/parallel_10y_profile_static.sh
#sh drivers/cpu-parallel_static_vary_p.sh
