#!/usr/bin/env python
# coding: utf-8

# # Testing ML Approaches on Data

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
from datetime import datetime

# # Configure GPU usage
# print("Checking GPU availability...")
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Enable memory growth to avoid allocating all GPU memory at once
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
#         print(f"TensorFlow will use GPU: {tf.test.is_gpu_available()}")
#     except RuntimeError as e:
#         print(f"GPU configuration error: {e}")
# else:
#     print("No GPUs found. Running on CPU.")

# # Set mixed precision for better GPU performance (optional but recommended)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train ML model with configurable batch size, model type, and number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 16)')
parser.add_argument('--model_type', type=str, default='default', help='Type of model to train (default: "default")')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training (default: 50)')
args = parser.parse_args()

# In[2]:


# Import the compressed data
dataset_name = 'four_params_Kbearing_c'
data_array = np.load(f'data/{dataset_name}.npy')
num_outputs = 1  # Change this based on the number of outputs in your dataset

# Shuffle the dataset rows with a known seed
np.random.seed(42)
np.random.shuffle(data_array)

# Decrease dataset size by factor of 10 for experimenting
# data_array = data_array[:len(data_array) // 1]

# Split the data into 80/20 train/test sets
train_data = data_array[:int(0.8 * len(data_array))]
test_data = data_array[int(0.8 * len(data_array)):]

# Split the features and labels
X_train = train_data[:, :-num_outputs]
y_train = train_data[:, -num_outputs:]
X_test = test_data[:, :-num_outputs]
y_test = test_data[:, -num_outputs:]

# Scale the features
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Scale the labels
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# In[3]:
print(f"Training with model type: {args.model_type}")

if args.model_type == "5_layer_DNN":
    # Build a simple neural network model for a single SIF prediction output layer
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(32),
        layers.LeakyReLU(),
        layers.Dense(num_outputs)  # Single output layer
    ])

elif args.model_type == "3_layer_DNN":
    # Build a simpler neural network model for a single SIF prediction output layer
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(32),
        layers.LeakyReLU(),
        layers.Dense(num_outputs)  # Single output layer
    ])

elif args.model_type == "default":
    # Build a very simple neural network model for a single SIF prediction output layer
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(32),
        layers.LeakyReLU(),
        layers.Dense(num_outputs)  # Single output layer
    ])

elif args.model_type == "ridge_regression":
    # Use tensorflow to create a linear regression model (ridge regression)
    lambda_value = 0.01  # Regularization strength
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],),),
        layers.Dense(num_outputs, kernel_regularizer=keras.regularizers.l2(lambda_value))  # Single output layer
    ])

elif args.model_type == "lasso_regression":
    # Use tensorflow to create a linear regression model (lasso regression)
    lambda_value = 0.01  # Regularization strength
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(num_outputs, kernel_regularizer=keras.regularizers.l1(lambda_value))  # Single output layer
    ])

# Compile the model, with Adam optimizer and a learning rate of 0.001
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')


# In[4]:


# Train the model with EarlyStopping
print(f"Training with batch size: {args.batch_size}, epochs: {args.num_epochs}")
model.fit(X_train_scaled, y_train_scaled, epochs=args.num_epochs, batch_size=args.batch_size, validation_split=0.2)#, callbacks=[early_stop])

# Save the model in the models/ directory with a filename containing the data name, model type, batch size, num_epochs, and timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/current/model_{dataset_name}_modeltype_{args.model_type}_batch{args.batch_size}_epochs{args.num_epochs}_{timestamp}.keras'
model.save(model_filename)
print(f"Model saved as: {model_filename}")

# Save the model history in the models directory
history_filename = f'models/current/history_{dataset_name}_modeltype_{args.model_type}_batch{args.batch_size}_epochs{args.num_epochs}_{timestamp}.pkl'
joblib.dump(model.history.history, history_filename)
print(f"Model history saved as: {history_filename}")

# Save the scaler objects to scale/unscale future data
joblib.dump(X_scaler, 'models/current/X_scaler.save')
joblib.dump(y_scaler, 'models/current/y_scaler.save')

