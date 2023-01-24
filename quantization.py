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

    