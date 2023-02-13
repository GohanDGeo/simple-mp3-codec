# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# demo2.py
#
# This demo file demonstrates the whole process of encoding and decoding a
# sound file, with the simple mp3 encoder/decoder described in this project.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from src.simple_mp3_codec import MP3codec
from src.helper import compression_ratio

# Load and define the required information for the filterbank
h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36
L = len(h)

# Load an audio file
filename = "myfile.wav"

try:
    samplerate, wavin = wavfile.read(filename)
except:
    print("Please give a valid file name!")
    exit()

wavin = np.array(wavin, dtype=float)

# Encode and decode the sound file
xhat, Ytot = MP3codec(wavin, h, M, N)

# Cast both arrays to int16 
wavin = np.int16(wavin)
xhat = np.int16(xhat)

# Save the decoded file to a .wav file
wavfile.write("xhat.wav", samplerate, xhat)

# Take into account the shift introduced during the analysis/synthesis steps
# As described, a shift of @(L-M) to the left is introduced
shift = L-M
wavin_shifted = np.copy(xhat[:-shift])
xhat_shifted = np.copy(wavin[shift:])

# Plot the error of the two files
error_fig = plt.figure()
plt.plot(xhat_shifted - wavin_shifted)
plt.title("Reconstructed - Original (Error)")
error_fig.show()

ax = plt.gca()
ax.set_ylim([np.min(wavin_shifted), np.max(wavin_shifted)])
ax.set_xlim([0, len(wavin_shifted)])

# Calculate the SNR
signal = np.mean(np.float64(wavin_shifted)**2)
noise = np.mean(np.float64(wavin_shifted-xhat_shifted)**2)

snr = 10*np.log10(signal/noise)
print(f"SNR: {snr}\n")

# Get the compression ratio between the original file
# and the produced huffman encoded bitstream
uncompressed_size, compressed_size, ratio = compression_ratio(filename, "bitstream.txt")
print(f"Original File Size: {uncompressed_size} KiloBytes\nEncoded File Size: {compressed_size} KiloBytes\nCompression Ratio: {ratio}\n")
plt.show()

