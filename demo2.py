import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from src.simple_mp3_codec import MP3codec

h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36

samplerate, wavin = wavfile.read("myfile.wav")

wavin = np.array(wavin, dtype=float)
xhat, Ytot = MP3codec(wavin, h, M, N)
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

