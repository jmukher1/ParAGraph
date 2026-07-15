#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-parallel-pa-3y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=16
cat /proc/cpuinfo
sh drivers/parallel_3y_static.sh