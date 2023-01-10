import numpy as np
from scipy.io import wavfile
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
import frame
import nothing

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

    buffer[L-M:] = data_pad[:frame_size]
    # For each frame, read points from the file
    for f in range(1, num_of_frames):

        buffer = np.roll(buffer, -(M*N))
        buffer[L-M:] = data_pad[f*frame_size:(f+1)*frame_size]
        # STEP (b)
        Y = frame.frame_sub_analysis(buffer, H, N)
        
        # STEP (c)
        Yc = nothing.donothing(Y)

        # Step (d)5
        Ytot.append(Yc)
    
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
        buffer = np.roll(buffer, -N,axis=0)
        buffer[int(-1 + L/M):, :] = Yh

        # STEP (f)
        z = frame.frame_sub_synthesis(buffer,G)

        xhat.append(z)

    xhat = np.asarray(xhat).ravel()

    return xhat

def codec0(wavin, h, M, N):
    Ytot = coder0(wavin, h, M, N)
    xhat = decoder0(Ytot, h, M, N)
    return xhat, Ytot