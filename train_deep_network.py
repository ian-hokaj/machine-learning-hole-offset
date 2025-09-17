import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Check GPU availability and print device info
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
else:
    print("No GPU detected. Training will use CPU.")

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
# Compile the model, with Adam optimizer and a learning rate of 0.001
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with EarlyStopping
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=64, validation_split=0.2)#, callbacks=[early_stop])

# Save the model in the models/ directory with a filename containing the data name and timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model.save(f'models/model_{dataset_name}_{timestamp}.keras')

# Save the scaler objects to scale/unscale future data
import joblib
joblib.dump(X_scaler, 'models/X_scaler.save')
joblib.dump(y_scaler, 'models/y_scaler.save')

# Test the model on the scaled test set
test_loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test Loss: {test_loss}")

# Construct appropriate plots for analysis
# Plot training, validation, and test loss
history = model.history.history
plt.plot(history['loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()