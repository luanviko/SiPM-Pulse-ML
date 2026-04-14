import tensorflow as tf
import numpy as np

# 1. Load the model and the scale
model = tf.keras.models.load_model('sipm_pulse_model_validated_pe.keras')
global_max = np.load("./data/global_max.npz")['global_max']
print(global_max)
N_events = 8000

# 2. Load the test waveforms
raw_waveforms = np.load("./data/waveforms_validated.npz")['waveforms'][N_events:]
baselines = np.load("./data/processed_data_validated.npz")['baseline'][N_events:]
waveforms_subtracted = np.zeros_like(raw_waveforms)
for i in range(len(raw_waveforms)):
    waveforms_subtracted[i] = raw_waveforms[i] - baselines[i]
A, B = 180, 400
X_test_clipped = waveforms_subtracted[:, A:B] / global_max
X_test_clipped = X_test_clipped[:, :, np.newaxis]  # Add channel
pe_value = 8.90 # mV/p.e., from 00b_pe_fitting.py
X_test_clipped = X_test_clipped / pe_value  # Normalize by PE value

# 3. Predict the normalized amplitudes
predictions_normalized = model.predict(X_test_clipped)
predicted_amps = predictions_normalized * global_max
np.savez("./data/predicted_amplitudes_validated_pe.npz", predicted_amps=predicted_amps)
print(predicted_amps.shape)