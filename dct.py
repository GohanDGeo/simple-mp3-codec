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

# Calculate the neighborhood of each discrete freq k
def Dksparse(Kmax):
    D = np.zeros((Kmax+1, Kmax+1))
    for k in range(Kmax+1):
        idx = []
        if 2 <= k or k < 282:
            idx = [k-2, k+2]
        elif 282 <= k or k < 570:
            n_range = np.arrange(2, 14)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
        elif 570 <= k or k < 1152:
            n_range = np.arrange(2, 27)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
            idx = idx[idx <= Kmax]
        
        if len(idx) > 0:
            D[k, idx] = 1
    
    return D