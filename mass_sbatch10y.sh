#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="10y-mass"
#SBATCH --account=zgdrasil
#SBATCH --partition=zgdrasil
#SBATCH --mem=200GB
#SBATCH --gres=gpu:A30

# Load necessary modules (e.g., CUDA)
module load cuda/12.6

./drivers/mass_10y_vary_p.sh 
