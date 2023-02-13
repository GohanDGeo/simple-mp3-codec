# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# quantization.py
#
# This file contains the functions for the quantizer and dequantizer

# Imports
import numpy as np
from .helper import discrete2Hz

# This function groups each discrete frequency [0, ..., K-1], to a critical band
# according to the given table
def critical_bands(K):
    
    # Initialize an array that holds each frequency's critibal band
    cb = np.zeros(K)

    # Get upper limits for each band
    upper_limits = [100, 200, 300, 400, 510, 630,
    770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700,
    3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

    # Initialize a counter showing which band is currently active
    current_band = 0
    
    # For each frequency
    for k in range(K):
        
        # Get frequency from discrete freq
        f = discrete2Hz(k)

        # If f is greater than the current band's limit
        # increase the current band number (do not increase if the last band is reached)
        
        if current_band < 24 and f > upper_limits[current_band]:
            current_band = min(current_band+1, 24)

        # Set the kth band
        cb[k] = current_band

    return cb

# Function that scales the DCT coefficients as described in the project
def DCT_band_scale(c):

    # Get number of coefficients
    K = len(c)
    
    # Get critical bands
    cb = critical_bands(K)

    # Get |c(i)|^(3/4)
    c_abs = np.power(np.abs(c), 3/4)

    # Get number of critical bands
    num_of_bands = len(np.unique(cb))

    # Initialize sc, cs
    sc = np.zeros(num_of_bands)
    cs = np.zeros(K)
    
    # For each band, find Sc(band), and calculate cs(i) for each i in band b
    for b in range(25):
        idx = cb == b
        # Check if there are freqs in this critical band
        if np.sum(idx) > 0:
            sc[b] = c_abs[idx].max()
            cs[idx] = np.sign(c[idx]) * (c_abs[idx] / sc[b])

    return cs, sc

# Function that implements a dead-zone quantizer, using @b bits
def quantizer(x, b):
    
    # Split [-1,1] 2^b + 1 zones
    d = np.linspace(-1, 1, (2**b)+1)

    # Remove the middle element which is zero
    d =np.delete(d, len(d)//2)

    # Initialize symbol vector
    symb_index = np.zeros(len(x))

    # For each element in x
    for i, s in enumerate(x):
        
        # Initialize level to last level
        level = len(d) - 2

        # If element is not rightmost decision boundary
        if s < d[-1]:

            # Check for each zone, if element is within its bounds
            for j in range(len(d)-1):
                dj = d[j]
                dj1 = d[j+1]

                if dj <= s < dj1:
                    level = j
                    break

        symb_index[i] = level
    
    # Set so the zone containing zero is indexed "0"
    symb_index -= (len(d)-1)//2

    return symb_index

# Function that dequantizes the quantized symbols, that were produced using the quantizer function
def dequantizer(symb_index, b):

    # Find decision boundaries, like in the quantizer
    d = np.linspace(-1, 1, (2**b)+1)
    d = np.delete(d, len(d)//2)
    
    # Find middle point for each zone
    mid_points = np.array([(d[i+1] + d[i])/2 for i in range(len(d)-1)])

    # Set quantized symbols as indices
    idx = (symb_index + len(d)//2 - 1).astype(int)

    # Get dequantized value
    xh = mid_points[idx]
    
    return xh

# Function that performs quantization for the coefficients of each band, while also finding the 
# best number of bits to use for each band, according to @Tg
def all_bands_quantizer(c, Tg):

    # The number of coefficients (in this implementation, always 1152)
    K = len(c)

    # Array that holds the critical band that each discrete freq belongs to
    cb = critical_bands(K)

    # The scaled coeffiecnts and scaling factors, as calculate from DCT_band_scale
    cs, SF = DCT_band_scale(c)

    # Initialize an arrya for the quantized symbols
    symb_index = np.zeros(K)

    # Initialize an array that holds the number of bits used for the quantization
    # of each band
    B = np.zeros(len(cb))

    # For each band
    for band in range(25):
        
        # Get coefficients belonging to this band
        idx = cb == band

        # Get c values of this band
        c_band = c[idx]
        
        # Get cs values of this band (Normalized)
        cs_band = cs[idx]

        # Get Tg values for c in this band
        Tgb = Tg[idx]

        # Initialize Pb to be larger than Tg (for each coefficient of the band)
        Pb = Tgb + 1

        b = 0

        symb_idx = []

        # Continue while Pb is higher than the threshold, increasing @b each time
        while np.any(Pb > Tgb):
            
            b += 1

            # Get symbol indices
            symb_idx = quantizer(cs_band, b)

            # Get dequantized c_tilde values
            c_d = dequantizer(symb_idx, b)
            
            # Get c_hat
            c_hat = np.sign(c_d) * np.power(np.abs(c_d) * SF[band], 4/3) 

            # Get error
            e_b = np.abs(c_band - c_hat)

            # Get error's power
            Pb = 10*np.log10(e_b**2)

        # Set the symbol indices to the array
        symb_index[idx] = symb_idx

        # Append the number of bits to the B array
        B[band] = b


    return symb_index, SF, B.astype(int)

# Function that dequantizes the quantized symbols
def all_bands_dequantizer(symb_index, B, SF):
    
    # The number of coefficients (in this implementation, always 1152)
    K = len(symb_index)
    
    # Array that holds the critical band that each discrete freq belongs to
    cb = critical_bands(K)
    
    # Initialize array that will hold the dequantized symbols
    xhat = np.zeros(K)

    # For each band
    for band in range(25):
        
        # Get coefficients of this band
        idx = cb == band

        # Get bits used in this band
        b = B[band]

        # Get symbol indices
        symb_idx = symb_index[idx]

        # Get dequantized c_tilde values
        c_d = dequantizer(symb_idx, b)
        
        # Get c_hat.NOTE: the absolute value of @c_d is used, to avoid RunTime warnings.
        # After all, since the exponent is 4/3 the sign does not matter in the final result.
        # (It is considered using the np.sign() function any way)
        c_hat = np.sign(c_d) * np.power(np.abs(c_d) * SF[band], 4/3) 

        xhat[idx] = c_hat

    return xhat