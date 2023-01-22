from scipy.fftpack import dct, idct
import numpy as np

def frameDCT(Y):
    c = []
    for i in range(Y.shape[1]):
        ci = dct(Y[:,i], type=4,  norm = 'ortho')
        c.append(ci)
    c = np.asarray(c).ravel()
    return c

def iframeDCT(c):
    N = 36
    M = 32
    Yh = np.zeros((N,M))
    for i in range(M):
        Yh[:,i] = idct(c[i*N:(i+1)*N], type=4,  norm = 'ortho')
    return Yh