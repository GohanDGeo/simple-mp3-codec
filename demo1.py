import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, rfft, irfft, fftshift
import sys
sys.path.insert(1, 'scripts and data') 
import mp3


h = np.load("scripts and data//h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
L = len(h)
H  = mp3.make_mp3_analysisfb(h, M)
G  = mp3.make_mp3_synthesisfb(h, M)

B = 689
fs = 44100
fft_size = 512
freqaxis = np.linspace(-fs/2, fs/2, fft_size)

# Plotting the amplitude spectrum of the filterbank
fft = np.fft.fft(H, axis=0)
filterbank_amp_spectrum = 10 * np.log10( np.abs(fft) **2)
plt.plot(freqaxis, filterbank_amp_spectrum)
plt.show()