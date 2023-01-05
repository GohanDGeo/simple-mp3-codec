import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#from scipy.fft import fft, rfft, irfft, fftshift
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
import frame

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

# coder0 
# STEP (a)
# Initialize frames
frames = []

# Get size of each frame
frame_size = (N-1)*M + L

# Reade wav file
samplerate, data = wavfile.read('scripts and data//myfile.wav')

# Calculate number of frames needed
num_of_frames = int(np.ceil(len(data)/frame_size))

# Pad with zeros
data_pad = np.pad(data, (0,len(data) % frame_size), 'constant')

# For each frame, read points from the file
for f in range(num_of_frames):
    buffer = data_pad[f*frame_size:(f+1)*frame_size]
    
    # STEP (b)
    res = frame.frame_sub_analysis(buffer, H, N)
    frames.append(res)






def coder0(wavin, h, M, N):
    pass

def decoder0(Ytot, h, M, N):
    pass