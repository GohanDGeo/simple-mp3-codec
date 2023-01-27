import numpy as np
from helper import discrete2Hz

def critical_bands(K):
    
    cb = np.zeros(K)

    # Get upper limits for each band
    upper_limits = [100, 200, 300, 400, 510, 630,
    770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700,
    3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

    # Initialize a counter showing which band is currently active
    current_band = 0
    
    for k in range(K):
        
        # Get frequency from discrete freq
        f = discrete2Hz(k)

        # If f is greater than the current band's limit
        # increase the current band number
        if f > upper_limits[current_band]:
            current_band = min(current_band+1, 24)

        # Set the kth band
        cb[k] = current_band
    
    # Plus one as it was zero indexed, and bands
    # are one indexed
    cb += 1

    return cb

def DCT_band_scale(c):

    # Get number of coefficients
    K = len(c)
    
    # Get critical bands
    cb = critical_bands(K)

    # Get |c(i)|^(3/4)
    c_abs = np.float_power(np.abs(c), 3/4)

    # Get number of critical bands
    num_of_bands = len(np.unique(cb))

    # Initialize sc, cs
    sc = np.zeros(num_of_bands)
    cs = np.zeros(K)
    
    # For each band, find Sc(band), and calculate cs(i) for each i in band b
    for b in range(num_of_bands):
        idx = cb == b
        sc[b] = c_abs[idx].max()
        cs[idx] = np.sign(c[idx]) * (c_abs[idx] / sc[b])

    return cs, sc

def quantizer(x, b):
    
    d = np.linspace(-1, 1, (2**b)+1)
    d =np.delete(d, len(d)//2)
    symb_index = np.digitize(x, d) - len(d)//2

    return symb_index

def dequantizer(symb_index, b):

    d = np.linspace(-1, 1, (2**b)+1)
    d =np.delete(d, len(d)//2)
    
    mid_points = np.array([(d[i+1] + d[i])/2 for i in range(len(d)-1)])
    idx = symb_index + len(d)//2 - 1
    
    xh = mid_points[idx]
    
    return xh

def all_bands_quantizer(c, Tg):

    K = len(c)

    cb = critical_bands(K)

    cs, SF = DCT_band_scale(c)

    symb_index = np.zeros(K)

    B = np.zeros(len(cb))

    for band in range(cb):
        
        # Get coefficients of this band
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

        # Continue while Pb is higher than the threshold
        while Pb > Tgb:

            b += 1

            # Get symbol indices
            symb_idx = quantizer(cs_band, b)

            # Get dequantized c_tilde values
            c_d = dequantizer(symb_idx, b)
            
            # Get c_hat
            c_hat = np.sign(c_d) * np.float_power(c_d * SF[band], 3/4) 

            # Get error
            e_b = np.abs(c_band - c_hat)

            # Get error's power
            Pb = 10*np.log10(e_b**2)

        # Set the symbol indices to the array
        symb_index[idx] = symb_idx

        # Append the number of bits to the B array
        B[band] = b


    return symb_index, SF, B

x = [-0.9, 0.9, 0.01]

symb_idx = quantizer(x, 3)
print(symb_idx)

xh = dequantizer(symb_idx, 3)
print(xh)


    