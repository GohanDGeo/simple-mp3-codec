import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#from scipy.fft import fft, rfft, irfft, fftshift
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
import frame
import nothing

from subband_filtering import coder0, decoder0, codec0

h = np.load("scripts and data//h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
L = len(h)
H  = mp3.make_mp3_analysisfb(h, M)
G  = mp3.make_mp3_synthesisfb(h, M)

B = 689
fs = 44100
fft_size = 512
freqaxis = np.linspace(-fs/2, fs/2, fft_size)

# Plotting the amplitude spectrum of the filterbank vs frequency
fft = np.fft.fft(H, axis=0)
filterbank_amp_spectrum = 10 * np.log10( np.abs(fft) **2)
fig = plt.figure()
plt.plot(freqaxis, filterbank_amp_spectrum)
fig.show()

# Plotting the amplitude spectrum of the filterbank vs barks
fig1 = plt.figure()
zaxis = 13*np.arctan(0.00076*freqaxis) + 3.5*np.arctan((freqaxis/7500)**2)
plt.plot(zaxis, filterbank_amp_spectrum)
fig1.show()

#plt.show()

samplerate, wavin = wavfile.read("scripts and data//myfile.wav")

xhat, Ytot = codec0(wavin, h, M, N)


wavfile.write("xhat.wav", samplerate, np.int16(xhat))

plt.plot(wavin - xhat)
plt.figure()

plt.plot(xhat)
plt.figure()

plt.plot(wavin)
plt.show()