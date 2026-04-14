import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def display_waveform(waveform, event, xscale=1, yscale=1, baseline=0.):
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(8,5))
    ax0.plot(np.arange(0,len(waveform))*xscale-baseline, waveform, "black")

pe_value = 8.90 # mV/p.e., from 00b_pe_fitting.py

# input_data = np.load("./data/raw_waveforms.npz")
input_data = np.load("./data/waveforms_validated.npz")
waveforms  = input_data['waveforms']
waveforms  = waveforms/pe_value

# input_data = np.load("./data/processed_data.npz")
input_data = np.load("./data/processed_data_validated.npz")
baseline = input_data['baseline']
amplitude = input_data['amplitude']

baseline = baseline/pe_value
amplitude = amplitude/pe_value  

# Normalize waveforms
for i in range(waveforms.shape[0]):
    waveforms[i] = waveforms[i] - baseline[i]
global_max = np.max(waveforms)
global_max = np.percentile(waveforms, 99.9)
# global_max = 1. 
normalized_waveforms = waveforms / global_max 
np.savez("./data/global_max.npz", global_max=global_max)

# Clip waveforms
A = 180
B = 400
clipped_waveforms = normalized_waveforms[:, A:B]
print(clipped_waveforms.shape)

# Create training dataset
# 20% of 40000 evets = 8000 events for testing 
N_events = 8000
Y_train = amplitude[:N_events] / global_max
Y_train = Y_train.reshape(-1, 1) 
X_train = clipped_waveforms[:N_events, :, np.newaxis]

# 1. Build model 
def build_sipm_model(input_shape):
    model = models.Sequential([
        # Layer 1: Local Feature Extraction
        # Finds the 'edges' and sharp rises of the pulse
        layers.Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        # Layer 2: Shape Recognition
        # Identifies the exponential decay profile
        layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        layers.GlobalAveragePooling1D(), # Condenses the waveform into high-level features
        
        # Regression Head
        # Maps those features to a single number: the Height
        layers.Dense(64, activation='leaky_relu'),
        layers.Dense(32, activation='leaky_relu'),
        layers.Dense(1, activation='linear') # 'linear' for regression (predicting a value)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_sipm_model((X_train.shape[1], 1))
model.summary()

# 2. Setup Early Stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 3. Train the model
print("Starting training...")
history = model.fit(
    X_train, Y_train,
    epochs=100,             # Maximum limit
    batch_size=32,          # Standard for this data size
    validation_split=0.2,   # Uses 20% of your 8000 for internal testing
    callbacks=[early_stop]
)

# 4. Save the trained model
model.save('sipm_pulse_model_validated_pe.keras')
print("Model saved to disk.")

# 5. Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title('Model Convergence')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()