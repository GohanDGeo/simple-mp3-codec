import numpy as np
from scipy.io import wavfile
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
import frame
from dct import *
from quantization import *
from rle import *
from huffman import *
from psychoacoustic import *

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
    data_pad = np.pad(wavin, (0, frame_size*num_of_frames - len(wavin) + L - M), 'constant')
    buffer = np.zeros((N-1)*M + L)

    Tq = np.load("scripts and data//Tq.npy", allow_pickle=True)[0]
    D = Dksparse(frame_size)

    # For each frame, read points from the file
    for f in range(num_of_frames):
        buffer = data_pad[f*frame_size:(f+1)*frame_size + L - M]
        # STEP (b)
        Y = frame.frame_sub_analysis(buffer, H, N)

        # STEP (c)
        Yc = frameDCT(Y)

        Tg = psycho(Yc, D, Tq) - 20

        symb_index, SF, B = all_bands_quantizer(Yc, Tg) 
        run_symbols = RLE(symb_index)

        frame_stream, frame_symbol_prob = huff(run_symbols)

        # Step (d)
        frame_info = dict()
        frame_info['SF'] = SF
        frame_info['B'] = B
        frame_info['frame_stream'] = frame_stream
        frame_info['frame_symbol_prob'] = frame_symbol_prob
        Ytot.append(frame_info)

    return Ytot

def decoder0(Ytot, h, M, N):
    
    G  = mp3.make_mp3_synthesisfb(h, M)
    L = G.shape[0]

    xhat = []
    
    Yhtot = []
    size = int(np.ceil(N-1 + L/M))
    
    buffer = np.zeros((size, M))
    K = N*M
    for frame_info in Ytot:

        # STEP (e)
        SF = frame_info['SF']
        B = frame_info['B']

        frame_stream = frame_info['frame_stream']
        frame_symbol_prob = frame_info['frame_symbol_prob']

        run_symbols = ihuff(frame_stream, frame_symbol_prob)
        symb_index = IRLE(run_symbols, K)
        Yc = all_bands_dequantizer(symb_index, B, SF)
        Yh = iframeDCT(Yc)
        Yhtot.append(Yh)
    
    Yhtot = np.vstack(Yhtot)
    Yhtot = np.pad(Yhtot, ((0,int(L/M)), (0,0)), 'constant')
    for f in range(len(Ytot)):
        buffer = Yhtot[f*N:(f+1)*N + int(L/M), :]

        # STEP (f)
        z = frame.frame_sub_synthesis(buffer,G)

        xhat.append(z)

    xhat = np.asarray(xhat).ravel()

    return xhat

def codec0(wavin, h, M, N):
    Ytot = coder0(wavin, h, M, N)
    xhat = decoder0(Ytot, h, M, N)
    return xhat, Ytot