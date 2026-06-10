#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-parallel-pa-10-30"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=16
sh drivers/cpu-parallel_static_vary_p.sh
