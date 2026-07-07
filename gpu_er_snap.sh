#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="snap-ERp"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1

# Load necessary modules (e.g., CUDA)
module load cuda/12.6

./drivers/p0001gpu_er_static_vary_p_snap.sh

./drivers/p0003gpu_er_static_vary_p_snap.sh

./drivers/p0005gpu_er_static_vary_p_snap.sh

./drivers/p001gpu_er_static_vary_p_snap.sh 


