#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="3y-p005-ERp"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --mem=127GB
#SBATCH --gres=gpu:h100:1

# Load necessary modules (e.g., CUDA)
module load cuda/12.6

./drivers/p005gpu_er_static_vary_p.sh 
