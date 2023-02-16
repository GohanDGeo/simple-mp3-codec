# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# subband_filtering0.py
#
# This file contains the main functions for the first task of the project,
# which just performs the analysis and synthesis steps for the frames, without
# any processing

# Imports
import numpy as np
import mp3
import frame
import nothing

# Performs the analysis step for a given wav file
def coder(wavin, h, M, N):

    # Create the H matrix from the given an impulse response @h and the number
    # of subbands @M
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
    # First term of padding is @(frame_size*num_of_frames - len(wavin)), to ensure
    # that the file is at long as the number of frames needed (as we take the ceil operation
    # when calculating the needed frames)
    # Then, @(L-M) zeros are added, for the buffer needed to get the last frame of the file
    data_pad = np.pad(wavin, (0, frame_size*num_of_frames - len(wavin) + L - M), 'constant')
    
    # Initialize a buffer which has dimensions @(N*M - (L-M)). 
    # The term @(L-M) is the overlap between two adjacent frames.
    buffer = np.zeros((N-1)*M + L)

    # For each frame, read samples from the file
    for f in range(num_of_frames):

        # Get the next @(N*M) samples, with an overlap of @(L-M) with the 
        # previous frame. The first frame has no overlap, thus the first
        # @(L-M) samples are lost.
        buffer = data_pad[f*frame_size:(f+1)*frame_size + L - M]

        # STEP (b)
        Y = frame.frame_sub_analysis(buffer, H, N)
        
        # STEP (c)
        Yc = nothing.donothing(Y)

        # Step (d)
        Ytot.append(Yc)

    return Ytot

# Performs the synthesis of the analyzed file, given @Ytot, which contains
# the "processed" frames (at this stage, no processing is done)
def decoder(Ytot, h, M, N):
    
    # Get the synthesis matrix
    G  = mp3.make_mp3_synthesisfb(h, M)
    L = G.shape[0]

    # Initialize a list which will contain the reconstructed file
    xhat = []
    
    # Initialize a list which will hold each decoded frame
    Yhtot = []

    # Set the height of the buffer, which is @(N + L/M), @N being the the height
    # of each frame, and @(L/M) the overlap factor
    size = int(np.ceil(N + L/M))
    
    # Initialize the buffer. It will have dimensions (@size, @M)
    buffer = np.zeros((size, M))

    for Yc in Ytot:

        # STEP (e)
        Yh = nothing.idonothing(Yc)
        Yhtot.append(Yh)
    
    # Since @Yhtot contains a list of frames, they now have to be 
    # vertically stacked to synthesize the reconstructed sound file
    Yhtot = np.vstack(Yhtot)
    # Pad @Yhtot, so the last frame has an extra @(L/M) under it.
    # This means that @(L-M) zeros will be added to the end of the 
    # reconstructed file. Since during encoding the first @(L-M) samples
    # are lost, the final reconstructed file will have the same
    # number of samples as the original file!
    Yhtot = np.pad(Yhtot, ((0,int(L/M)), (0,0)), 'constant')

    # For each frame
    for f in range(len(Ytot)):
        # Fill the buffer with @(N + L/M) rows, 
        # with @(L/M) overlapping with the previous frame
        buffer = Yhtot[f*N:(f+1)*N + int(L/M), :]

        # STEP (f)
        # Synthesize the samples
        z = frame.frame_sub_synthesis(buffer,G)

        # Append the reconstructed samples to the list
        xhat.append(z)

    # Make the list into a numpy array
    xhat = np.asarray(xhat).ravel()

    return xhat

# Perform both coder0, decoder0 steps
# and return a reconstructed @xhat array
def codec(wavin, h, M, N):
    Ytot = coder(wavin, h, M, N)
    xhat = decoder(Ytot, h, M, N)
    return xhat, Ytot