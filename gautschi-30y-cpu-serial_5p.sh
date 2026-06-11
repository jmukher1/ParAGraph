#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-s-pa-30_5p"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=1
sh drivers/serial_30y_profile_static_5p.sh
