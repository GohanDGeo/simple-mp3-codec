import numpy as np
from scipy.io import wavfile
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
import frame
import nothing

class InputBuffer:
  
    def __init__(self, size, type='float32'):
        self.size = size
        self.pos  = 0
        self.samples = np.zeros(size, dtype=type)

    def insert(self, data_in):
        length = len(data_in)
        if self.pos + length <= self.size:
            self.samples[self.pos:self.pos+length] = data_in
        else:
            overhead = length - (self.size - self.pos)
            self.samples[self.pos:self.size] = data_in[:-overhead]
            self.samples[0:overhead] = data_in[-overhead:]
        self.pos += length
        self.pos %= self.size

class InputBuffer2D:
  
    def __init__(self, size1, size2, type='float32'):
        self.size1 = size1
        self.size2 = size2
        self.pos  = 0
        self.samples = np.zeros((size1, size2), dtype=type)

    def insert(self, data_in):
        length = data_in.shape[0]

        if self.pos + length <= self.size1:
            self.samples[self.pos:self.pos+length, :] = data_in
            #print(data_in.shape)
            #print(self.samples[self.pos:self.pos+length, :].shape)
        else:
            overhead = length - (self.size1 - self.pos)
            self.samples[self.pos:self.size1, :] = data_in[:-overhead, :]
            self.samples[0:overhead, :] = data_in[-overhead:, :]

        self.pos += length
        self.pos %= self.size1

def coder0(wavin, h, M, N):

    H  = mp3.make_mp3_analysisfb(h, M)
    
    # STEP (a)
    # Initialize frames
    Ytot = []

    # Get size of each frame
    L = H.shape[0]
    frame_size = N*M #(N-1)*M + L

    # Calculate number of frames needed
    num_of_frames = int(np.ceil(len(wavin)/frame_size))

    # Pad with zeros
    data_pad = np.pad(wavin, (0,len(wavin) % frame_size), 'constant')

    buffer = np.zeros((N-1)*M + L)
    print(buffer.shape)
    i = 0
    # For each frame, read points from the file
    for f in range(num_of_frames):

        buffer = np.roll(buffer, -(L-M))
        buffer[L-M:] = data_pad[f*frame_size:(f+1)*frame_size]
        # STEP (b)
        Y = frame.frame_sub_analysis(buffer, H, N)
        
        # STEP (c)
        Yc = nothing.donothing(Y)

        # Step (d)5
        Ytot.append(Yc)
        i += 1
        #if i==2:
         #   break
    
    # Return a list of frames. Each list is a 2D numpy array
    return Ytot

def decoder0(Ytot, h, M, N):
    
    G  = mp3.make_mp3_synthesisfb(h, M)
    L = G.shape[0]

    xhat = []
    size = int(np.ceil(N-1 + L/M))
    buffer = np.zeros((size, M))#InputBuffer2D(int(np.ceil(N-1 + L/M)), M)
    for Yc in Ytot:

        # STEP (e)
        Yh = nothing.idonothing(Yc)
        buffer = np.roll(buffer, -int(-1 + L/M),axis=0)
        buffer[int(-1 + L/M):, :] = Yh

        # STEP (f)
        z = frame.frame_sub_synthesis(buffer,G)

        xhat.append(z)

    print(np.asarray(xhat).shape)
    xhat = np.asarray(xhat).ravel()

    return xhat

def codec0(wavin, h, M, N):
    Ytot = coder0(wavin, h, M, N)
    xhat = decoder0(Ytot, h, M, N)
    return xhat, Ytot