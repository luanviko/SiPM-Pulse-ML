import numpy as np
import json, sys

def split_preamble(preamble):
    '''
    Docstring for split_preamble
    
    :param preamble: A string containing the oscilloscope parameters such as mV/div, time/div, etc.
    :return: A dictionary with the oscilloscope parameters as key-value pairs.
    '''
    
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
    '''
    Find baseline, global maximum, area, and position 
    of the global maximum for each waveform.
    
    :param waveforms: Description
    :param A: Area integration start index (if None, starts from 0)
    :param B: Area integration end index (if None, ends at the last index of the waveform)
    :return: baseline, amplitude, area, STD_position

    NOTE: STD stands for Simple Threshold Discriminator, which in 
    this context is the sample index of the global maximum in the waveform.
    '''

    N_entries    = waveforms.shape[0]
    baseline     = np.zeros(N_entries)
    amplitude    = np.zeros(N_entries)
    area         = np.zeros(N_entries)
    STD_position = np.zeros(N_entries)

    if A == None: A = 0
    if B == None: B = len(waveforms[0])
    
    for i in range(0, N_entries):
        baseline[i] = np.average(waveforms[i][0:100])
        rel_j_max = np.argmax(waveforms[i][A:B])
        j_max = rel_j_max + A
        area[i] = np.sum(waveforms[i][A:B]-baseline[i])
        STD_position[i] = j_max
        amplitude[i] = waveforms[i][j_max]-baseline[i]

    return baseline, amplitude, area, STD_position

# 1. Load the raw waveforms and the preamble
print("Loading raw waveforms and preamble...")
input_data = np.load("./data/00_raw_waveforms.npz")

# 2. Break preamble into a dictionary for easy access to parameters
preamble = split_preamble(input_data['preamble'].item())

# 3. Convert raw waveforms to physical units
#    - Convert from ADC counts to mV using YMUlt and YZEro
#    - Convert time axis from sample indices to ns using XINcr and XZEro
waveforms  = input_data['waveforms']
waveforms = waveforms*preamble['YMUlt'] + preamble["YZEro"]
waveforms = waveforms*1.e3
time_axis = np.arange(waveforms.shape[1]) * preamble['XINcr'] + preamble['XZEro']
time_axis = time_axis * 1.e9

# 4. Analyze the waveforms to extract baseline, amplitude, area, and STD_position
print("Analyzing waveforms to extract pulse information...")
baseline, amplitude, area, STD_position = analyze_data(waveforms, A = 180, B=400)

# 5. Apply validation criteria to filter out invalid events
print("Validating events: Removing noisy waveforms or inappropriate timing...")
amp_mask = amplitude > 0
time_mask = (STD_position > 180) & (STD_position < 400)
mask = amp_mask & time_mask
valid_waveforms = waveforms[mask]

# 6. Save the validated waveforms
print("Saving validated waveforms...")
np.savez_compressed(
    "./data/01_validated_waveforms.npz", 
    waveforms=valid_waveforms, 
    time_axis=time_axis, 
    preamble=preamble
    )

# 7. Save the extracted parameters for the validated events
print("Saving pulse information...")
np.savez_compressed(
    "./data/02_pulse_information.npz", 
    baseline=baseline[mask], 
    amplitude=amplitude[mask], 
    area=area[mask], 
    STD_position=STD_position[mask]
    )

print("Done!")