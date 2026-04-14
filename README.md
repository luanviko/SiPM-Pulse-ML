# SiPM Pulse Analysis with CNN

This is a small project to try and predict the amplitude of photosensor pulses using a Convolutional Neural Network using Python and Tensorflow. 

Data visualization is implemented in a dashboard based in Plotly and Streamlit.

In the following, I will use the words SiPM and photosensor interchangeably. 

\[This file is a draft and more details will be added as I improve the model and dashboard\]

## Experimental Setup

* An LED generates pulses at 470 nm, at a frequency of about 1 MHz.

* These light pulses are captured by an optical fibre and guided to a Silicon Photomultiplier (SiPM) photosensor.

* The photosensor generates electric pulses, whose waveforms are acquired with an oscilloscope.


## Data Workflow

```
Data: Waveforms -> Signal Processing -> Amplitude Extraction -> Training -> Prediction -> Comparison
```

```
Python: SCPI -> Numpy -> Tensorflow -> Streamlit & Plotly 
```

```
Preliminary visualization: Matplotlib and pyRoot.
```

```
Dashboard: Streamlit and Plotly.
```

### Training

* 8000 waveforms used for training.
* Convert from ADC and sample number to mV and ns.
* Select waveforms with positive amplitude and within a 
* Clip waveforms to interested time window. 
* Normalize waveforms. 
* Training waveforms to their values in p.e.

### Predicting

* More than 30000 waveforms used for prediction.




