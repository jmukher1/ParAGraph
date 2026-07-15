#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="cpu-sn3y"
#SBATCH --account=ayg
#SBATCH --partition=cpu
#SBATCH --qos=standby


source ~/.bashrc

cat /proc/cpuinfo

export OMP_NUM_THREADS=16
sh drivers/parallel_3y_profile_static_snap.sh
#sh drivers/parallel_3y_profile_static.sh
#sh drivers/cpu-parallel_static_vary_p.sh
