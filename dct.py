# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# dct.py
# 
# This file contains the function related to the DCT processing of an mp3 frame

# Imports
from scipy.fftpack import dct, idct
import numpy as np

# Take a frame @Y of size NxM and find the DCT coefficients
# The coefficients are calculate along each column (each subband)
# The final vector @c has length NM, as it has N coefficients for each of the M subbands
def frameDCT(Y):
    c = []
    for i in range(Y.shape[1]):
        ci = dct(Y[:,i], type=2,  norm = 'ortho')
        c.append(ci)
    c = np.asarray(c).ravel()
    return c

# For a vector of coefficients @c, calculate the inverse DCT transform,
# recreating a frame of size NxM.
# This means that every N coefficients, a column (a subband) is re-created.
def iframeDCT(c):
    N = 36
    M = 32
    Yh = np.zeros((N,M))
    for i in range(M):
        Yh[:,i] = idct(c[i*N:(i+1)*N], type=2,  norm = 'ortho')
    return Yh

# Calculate the power of each DCT coefficient in dB
def DCTpower(c):
    return 10*np.log10(np.abs(c)**2)