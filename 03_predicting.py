import tensorflow as tf
import numpy as np

# 1. Load the model and the scale
model = tf.keras.models.load_model('./data/04_sipm_model.keras')
global_max = np.load("./data/03_global_max.npz")['global_max']
N_events = 8000

# 2. Load the test waveforms
raw_waveforms = np.load("./data/01_validated_waveforms.npz")['waveforms'][N_events:]
baselines = np.load("./data/02_pulse_information.npz")['baseline'][N_events:]
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
np.savez("./data/05_predicted_amplitudes.npz", predicted_amps=predicted_amps)
print(predicted_amps.shape)