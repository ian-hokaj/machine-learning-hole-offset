#!/bin/bash

# Clear the logs/current/ and models/current/ directory before running new jobs
# rm -rf logs/current/*
# rm -rf models/current/*
# mkdir -p logs/current
# mkdir -p models/current

# Slurm sbatch options for job array with GPU
#SBATCH -o logs/current/jobArray_experiment-%A-%a.out
# #SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH -a 0-4

# Loading the required module
source /etc/profile
module load anaconda/2023a-tensorflow

# Default batch sizes to experiment with
num_epochs=5
batch_sizes=(8 16 32 64 128)
model_names=("default" "ridge_regression" "lasso_regression" "3_layer_DNN" "5_layer_DNN")

# Get the batch size for this array task
batch_size=${batch_sizes[$SLURM_ARRAY_TASK_ID]}
model_name=${model_names[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
# echo "Training with batch size: $batch_size"
echo "Model type: $model_name"

# Run the training
# python model_experimentation.py --batch_size $batch_size --model_type $model_name
python model_experimentation.py --model_type $model_name --num_epochs $num_epochs