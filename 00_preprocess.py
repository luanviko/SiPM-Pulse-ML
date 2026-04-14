import numpy as np
import json, sys

def split_preamble(preamble):
    print(preamble)
    preamble = preamble.strip()
    preamble = preamble.replace('{', '')
    preamble = preamble.replace('}', '')
    preamble = preamble.replace('"', '')
    preamble = preamble.replace('n', '')
    preamble = preamble.replace('\\', '')
    preamble = preamble.split(',')
    preamble_dict = {}
    for entry in preamble:
        key, value = entry.split(':')
        key = key.strip().replace('"', '')
        try:
            value = np.float64(value.strip().replace('"', ''))
        except:
            value = value.strip().replace('"', '')
        preamble_dict[key] = value
    return preamble_dict

def analyze_data(waveforms, A = None, B = None):

    N_entries    = waveforms.shape[0]
    baseline     = np.zeros(N_entries)
    amplitude    = np.zeros(N_entries)
    area         = np.zeros(N_entries)
    STD_position = np.zeros(N_entries)

    if A == None: A = 0
    if B == None: B = len(waveforms[0])
    
    print(N_entries)
    for i in range(0, N_entries):
        baseline[i] = np.average(waveforms[i][0:100])
        rel_j_max = np.argmax(waveforms[i][A:B])
        j_max = rel_j_max + A
        area[i] = np.sum(waveforms[i][A:B]-baseline[i])
        STD_position[i] = j_max
        amplitude[i] = waveforms[i][j_max]-baseline[i]

    return baseline, amplitude, area, STD_position

# 1. Load the raw waveforms and the preamble
input_data = np.load("./data/raw_waveforms.npz")

# 2. Process the waveforms and extract features
preamble = split_preamble(input_data['preamble'].item())

# 3. Convert raw waveforms to physical units
waveforms  = input_data['waveforms']
waveforms = waveforms*preamble['YMUlt'] + preamble["YZEro"]
waveforms = waveforms*1.e3
time_axis = np.arange(waveforms.shape[1]) * preamble['XINcr'] + preamble['XZEro']
time_axis = time_axis * 1.e9

# 4. Analyze the waveforms to extract baseline, amplitude, area, and STD_position
baseline, amplitude, area, STD_position = analyze_data(waveforms, A = 180, B=400)

# 5. Apply validation criteria to filter out invalid events
amp_mask = amplitude > 0
time_mask = (STD_position > 180) & (STD_position < 400)
mask = amp_mask & time_mask
valid_waveforms = waveforms[mask]

# 6. Save the validated waveforms and their corresponding features
np.savez("./data/waveforms_validated.npz", waveforms=valid_waveforms, time_axis=time_axis, preamble=preamble)
np.savez("./data/processed_data_validated.npz", baseline=baseline[mask], amplitude=amplitude[mask], area=area[mask], STD_position=STD_position[mask])
