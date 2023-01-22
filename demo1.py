import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
from subband_filtering import codec0

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

samplerate, wavin = wavfile.read("scripts and data//myfile.wav")

wavin = np.array(wavin, dtype=float)
xhat, Ytot = codec0(wavin, h, M, N)
wavfile.write("xhat.wav", samplerate, np.int16(xhat))


wavin = np.int16(wavin)
xhat = np.int16(xhat)

shift = 480
wavin_shifted = np.copy(xhat[:-shift])
xhat_shifted = np.copy(wavin[shift:])

fig2 = plt.figure()
plt.plot(xhat_shifted - wavin_shifted)
fig2.show()

ax = plt.gca()
ax.set_ylim([np.min(wavin_shifted), np.max(wavin_shifted)])

signal = np.mean(np.float64(wavin_shifted)**2)
noise = np.mean(np.float64(wavin_shifted-xhat_shifted)**2)

snr = 10*np.log10(signal/noise)
print(f"SNR {snr}")

plt.show()

