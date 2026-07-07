#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="all_snap_ERp"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1

# Load necessary modules (e.g., CUDA)
module load cuda/12.6

./drivers/mass_cuda_er_vary-p_snap.sh
