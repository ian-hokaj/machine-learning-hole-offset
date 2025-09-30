#!/bin/bash

# Slurm sbatch options for job array with GPU
#SBATCH -o logs/current/single_experiment-%A-%a.out
# #SBATCH --gres=gpu:volta:1
#SBATCH -c 4

# Loading the required module
source /etc/profile
module load anaconda/2023a-tensorflow

# Run the training
python model_experimentation.py