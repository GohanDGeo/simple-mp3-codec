import numpy as np

# Calculates barks from frequency
def Hz2Barks(f):
    z = 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)
    return z

def discrete2Hz(k):
    B = 689
    N = 36
    return k * B / N