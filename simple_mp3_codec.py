# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# simple_mp3_codec.py
#
# This file contains the main functions for the mp3 coder/decoder/codec

# Imports
import numpy as np
import mp3
import frame
from dct import *
from quantization import *
from rle import *
from huffman import *
from psychoacoustic import *

# Performs the encoding step for a given wav file
def MP3cod(wavin, h, M, N):

    # Create the H matrix from the given an impulse response @h and the number
    # of subbands @M
    H  = mp3.make_mp3_analysisfb(h, M)
    
    # STEP (a)
    # Initialize frames
    Ytot = []

    # Get size of each frame
    L = H.shape[0]
    frame_size = N*M

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

    # Load the @Tq thresholds
    Tq = np.load("Tq.npy", allow_pickle=True)[0]
    
    # Create the @D sparse matrix
    D = Dksparse(frame_size)

    # Create a .txt file to save the resulting encoding
    bitstream = open("bitstream.txt", 'w')

    # For each frame, read samples from the file
    for f in range(num_of_frames):

        # Get the next @(N*M) samples, with an overlap of @(L-M) with the 
        # next/previous frame. The first frame has no overlap on its first samples, 
        # thus the first @(L-M) samples are lost.
        buffer = data_pad[f*frame_size:(f+1)*frame_size + L - M]

        # STEP (b)
        # Get the frame
        Y = frame.frame_sub_analysis(buffer, H, N)

        # STEP (c)
        # Get the DCT coefficients
        Yc = frameDCT(Y)

        # Apply the psychoacoustic model, and move it by 20
        # 20 was found with listening tests, while keeping a decent
        # compression ratio
        Tg = psycho(Yc, D, Tq) - 20

        # Quantize the coefficients
        symb_index, SF, B = all_bands_quantizer(Yc, Tg) 

        # Perform RLE encoding
        run_symbols = RLE(symb_index)

        # Perform huffman encoding
        frame_stream, frame_symbol_prob = huff(run_symbols)

        # And append the frames bitstream to the file
        bitstream.write(frame_stream)

        # Step (d)
        # Save the necessary information for decoding the frame to a dictionary

        # Initialize the dictionary
        frame_info = dict()
        # Save the scaling factors
        frame_info['SF'] = SF
        # Save the number of bits used for quantizing each band
        frame_info['B'] = B
        # Save the number of bits produced from the huffman encoding in this frame
        frame_info['frame_stream_bits'] = len(frame_stream)
        # Save the symbol probabilities for each symbol (in the frame)
        frame_info['frame_symbol_prob'] = frame_symbol_prob
        # Append the dictionary to @Ytot, which will contain decoding information for all frames
        Ytot.append(frame_info)

    # Close the .txt file
    bitstream.close()

    # Return @Ytot, which contains the decoding information
    return Ytot

# Performs the decoding of the encoded file, given @Ytot, which contains
# the decoding information for each frame
def MP3decod(Ytot, h, M, N):
    
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
    
    # Number of samples in each frame
    K = N*M

    # Open the bitstream file, to read the huffman encoded bits
    bitstream = open("bitstream.txt", 'r')
    # Decode each frame, thus read the decoding information for each frame
    # in each iteration
    for frame_info in Ytot:

        # STEP (e)
        # Get the scaling factors for this frame
        # as well as the quantization bits for this frame
        SF = frame_info['SF']
        B = frame_info['B']

        # Get number of bits to read in this frame
        frame_stream_bits = frame_info['frame_stream_bits']
        # And read these bits from the file
        frame_stream = bitstream.read(frame_stream_bits)

        # Get the symbol probabilites for this frame
        frame_symbol_prob = frame_info['frame_symbol_prob']

        # Decode the huffman bitstream
        run_symbols = ihuff(frame_stream, frame_symbol_prob)

        # Decode the run-length encoding
        symb_index = IRLE(run_symbols, K)

        # Dequantize the DCT coefficients
        Yc = all_bands_dequantizer(symb_index, B, SF)

        # And perform the inverse DCT to get the frame samples
        Yh = iframeDCT(Yc)

        # Append the decoded frame to @Yhtot
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
        # with @(L/M) overlapping with the next/previous frame
        buffer = Yhtot[f*N:(f+1)*N + int(L/M), :]

        # STEP (f)
        # Synthesize the samples
        z = frame.frame_sub_synthesis(buffer,G)

        # Append the reconstructed samples to the list
        xhat.append(z)

    # Make the list an array and return the decoded, reconstructed
    # sound array!
    xhat = np.asarray(xhat).ravel()

    return xhat

# Performs the coder and decoder steps
# Given a file @wavin (in numpy format),
# an encoded bitstream will be produced, as well
# as decoded array will be returned (@xhat)
def MP3codec(wavin, h, M, N):
    Ytot = MP3cod(wavin, h, M, N)
    xhat = MP3decod(Ytot, h, M, N)
    return xhat, Ytot