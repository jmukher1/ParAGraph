#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-10y"
#SBATCH --account=ayg
#SBATCH --partition=cpu
#SBATCH --qos=standby


source ~/.bashrc

cat /proc/cpuinfo

export OMP_NUM_THREADS=16
sh drivers/cpu-parallel_pa_10y_static_multirun.sh
