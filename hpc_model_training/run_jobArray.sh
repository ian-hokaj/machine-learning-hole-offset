#!/bin/bash

# ISSUE: runs on login node, gets in the way of spinning up the tasks. Need to be in a separate script before this bash script
# Clear the logs/current/ and models/current/ directory before running new jobs
# rm -rf logs/current/*
# rm -rf models/current/*
# mkdir -p logs/current
# mkdir -p models/current

# Slurm sbatch options for job array with GPU
#SBATCH -o logs/current/jobArray_experiment-%A-%a.out
# #SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH -a 0-3

# Loading the required module
source /etc/profile
module load conda/Python-ML-2025b-tensorflow
# module load anaconda/2023a-tensorflow

# Default parameters to experiment with
dataset_name=("major_params_Kbearing_c")
num_epochs=(500)
batch_sizes=(16)
model_names=("4_layer_ridge_regression" "4_layer_lasso_regression")
regularization_strengths=(0.000001 0.00001)

# Calculate total number of parameter combinations
num_batch_sizes=${#batch_sizes[@]}
num_models=${#model_names[@]}
num_reg_strengths=${#regularization_strengths[@]}
total_combinations=$((num_batch_sizes * num_models * num_reg_strengths))

echo "Total parameter combinations: $total_combinations"
echo "Current task ID: $SLURM_ARRAY_TASK_ID"

# Calculate which combination this task should run
# We'll iterate through: batch_size (outer) -> model (middle) -> reg_strength (inner)
batch_idx=$((SLURM_ARRAY_TASK_ID / (num_models * num_reg_strengths)))
remaining=$((SLURM_ARRAY_TASK_ID % (num_models * num_reg_strengths)))
model_idx=$((remaining / num_reg_strengths))
reg_idx=$((remaining % num_reg_strengths))

# Get the actual parameter values for this combination
batch_size=${batch_sizes[$batch_idx]}
model_name=${model_names[$model_idx]}
reg_strength=${regularization_strengths[$reg_idx]}

echo "Parameter combination:"
echo "  Batch size: $batch_size"
echo "  Model type: $model_name"
echo "  Regularization strength: $reg_strength"

# Run the training with all parameter combinations
python model_experimentation.py --data_file $dataset_name --batch_size $batch_size --model_type $model_name --num_epochs $num_epochs --reg_strength $reg_strength