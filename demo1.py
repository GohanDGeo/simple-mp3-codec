# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# demo1.py
#
# This demo file demonstrates the first task of this project (Section 3.1 Subband Filtering)
# Firstly, the impulse response of the filter is plotted, along with plots for the magnitude of the filtarbank
# vs frequency and vs barks.
# 
# Given an audio file, the file is encoded and decoded (without processing the frames, using the nothing functions)
# and the difference between the original and reconstructed files is plotted. 
# NOTE: The reconstructed file, loses the first L-M (here 480) first samples of the original file (due to no overlapping for the
# first frame). This means that xhat starts from sample 480 of the original file, and that the last 480 samples are 0.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import mp3
from subband_filtering0 import codec

# Load and define the required information for the filterbank
h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
L = len(h)

fs = 44100
fft_size = 512

# Create the analysis and synthesis matrices
H  = mp3.make_mp3_analysisfb(h, M)
G  = mp3.make_mp3_synthesisfb(h, M)

# Define the positive frequency axis
positive_freqaxis = np.linspace(0, fs/2, fft_size//2 + 1)

# Plot the impulse response @h
impulse_response_fig = plt.figure()
plt.plot(h)
plt.title("Impulse Response")
impulse_response_fig.show()

# Plot the amplitude spectrum of the filterbank vs frequency
fft = np.fft.rfft(H, axis=0)
filterbank_amp_spectrum = 10 * np.log10( np.abs(fft) **2)
filterbank_freq_fig = plt.figure()
plt.plot(positive_freqaxis, filterbank_amp_spectrum)
plt.title("Filterbank Magnitude VS Frequency")
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
filterbank_freq_fig.show()

# Plot the amplitude spectrum of the filterbank vs barks
filterbank_bark_fig = plt.figure()
zaxis = 13*np.arctan(0.00076*positive_freqaxis) + 3.5*np.arctan((positive_freqaxis/7500)**2)
plt.plot(zaxis, filterbank_amp_spectrum)
plt.title("Filterbank Magnitude VS Barks")
plt.ylabel("Magnitude (dB)")
plt.xlabel("Barks")
filterbank_bark_fig.show()

# Load an audio file 
filename = "myfile.wav"

try:
    samplerate, wavin = wavfile.read(filename)
except:
    print("Please give a valid file name!")
    exit()

wavin = np.array(wavin, dtype=float)

# Reconstruct the audio file
xhat, Ytot = codec(wavin, h, M, N)

# Cast to int16 so the audio file can be played by media players
wavin = np.int16(wavin)
xhat = np.int16(xhat)

# Save the reconstructed audio file
wavfile.write("reconstructed_file0.wav", samplerate, np.int16(xhat))

# Shif the audio files accordingly so the overlap
shift = L-M
wavin_shifted = np.copy(xhat[:-shift])
xhat_shifted = np.copy(wavin[shift:])

# Plot the error of the two files
error_fig = plt.figure()
plt.plot(xhat_shifted - wavin_shifted)
plt.title("Reconstructed - Original (Error)")
error_fig.show()

# And set the scale the same as the scale of the original file
ax = plt.gca()
ax.set_ylim([np.min(wavin_shifted), np.max(wavin_shifted)])

# Calculate the SNR
signal = np.mean(np.float64(wavin_shifted)**2)
noise = np.mean(np.float64(wavin_shifted-xhat_shifted)**2)

# Print the SNR between the two audio files
snr = 10*np.log10(signal/noise)
print(f"SNR: {snr}")

# Show plots
plt.show()
