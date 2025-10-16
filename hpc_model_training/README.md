# machine-learning-hole-offset
Attempt to use machine learning to approximate the Stress Intensity Factors (SIF, or K-solutions) for finite-width offset hole specimen geometries

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ian-hokaj/machine-learning-hole-offset.git
cd machine-learning-hole-offset
```

### 2. Running on HPC with Slurm

This project includes a parameter sweep script that runs multiple machine learning experiments in parallel using Slurm job arrays.

#### Prerequisites
- Access to an HPC cluster with Slurm scheduler
- TensorFlow/ML modules available on the cluster

#### Running the Parameter Sweep

1. **Clean up previous runs (optional)**:
   ```bash
   rm -rf logs/current/*
   rm -rf models/current/*
   mkdir -p logs/current
   mkdir -p models/current
   ```

2. **Submit the job array**:
   ```bash
   sbatch run_jobArray.sh
   ```

3. **Monitor job status**:
   ```bash
   squeue -u $USER
   ```

4. **Check outputs**:
   ```bash
   # View job logs
   ls logs/current/
   
   # Check a specific job output
   tail logs/current/jobArray_experiment-<JobID>-<TaskID>.out
   
   # View trained models
   ls models/current/
   ```

#### Parameter Sweep Details

The `run_jobArray.sh` script runs a comprehensive parameter sweep over:
- **Batch sizes**: 8, 16, 32, 64, 128
- **Model types**: 4_layer_ridge_regression, 4_layer_lasso_regression
- **Regularization strengths**: 0.000001, 0.00001, 0.0001, 0.001, 0.01
- **Total combinations**: 50 experiments (5 × 2 × 5)

Each experiment trains for 200 epochs and saves both the trained model and training history.

#### Customizing Parameters

To modify the parameter sweep, edit the arrays in `run_jobArray.sh`:
```bash
batch_sizes=(8 16 32 64 128)
model_names=("4_layer_ridge_regression" "4_layer_lasso_regression")
regularization_strengths=(0.000001 0.00001 0.0001 0.001 0.01)
```

Remember to update the `#SBATCH -a` range to match the total number of combinations.

