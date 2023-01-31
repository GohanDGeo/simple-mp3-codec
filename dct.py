from scipy.fftpack import dct, idct
from scipy.sparse import csr_matrix
import numpy as np
from helper import Hz2Barks, discrete2Hz

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
    D = np.zeros((Kmax, Kmax))
    for k in range(Kmax):
        idx = []
        if 2 <= k < 282:
            idx = [k-2, k+2]
        elif 282 <= k < 570:
            n_range = np.arange(2, 14)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
        elif 570 <= k < 1152:
            n_range = np.arange(2, 27)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
            idx = idx[idx < Kmax]
        
        if len(idx) > 0:
            D[k, idx] = 1
    
    return csr_matrix(D)

# Get tonal components
def STinit(c, D):
    # Get power of coefficients
    Pc = DCTpower(c)

    # Initialize ST list
    ST = []

    Dneighbors = np.split(D.indices, D.indptr[1:-1])
    # Check if it is a tonal component
    for k in range(len(c)):
        # Compare the power of the kth coefficient with each left and right coeff. and
        # with its Dk neighbors
        neighbors = np.array([k-1, k+1])

        # If k == 1151, it does not have a right neighbor, so only check the left and Dk
        neighbors = neighbors[neighbors < len(c)]
        neighbors = neighbors[neighbors >= 0]

        # Get a list of all powers the kth coefficient has to compare with
        #compare_pc = np.concatenate((Pc[neighbors], Pc[np.nonzero(D[k,:])] + 7))
        
        # And compare.

        Dk = Dneighbors[k]
        if np.all(Pc[k] > Pc[neighbors]) or np.all(Pc[k] > Pc[Dk] + 7):
            ST.append(k)

    ST = np.asarray(ST)
    return ST
    
# Calculate the power of the maskers
def MaskPower(c, ST):
    Pc = DCTpower(c)
    PM = np.zeros(len(ST))

    # For each masker
    for k in range(len(ST)):

        # Get the masker's neighbors
        neighbors = np.array([k-1, k, k+1])

        # Check if the neighbors exist (within bounds)
        neighbors = neighbors[neighbors < len(c)]
        neighbors = neighbors[neighbors >= 0]

        # Get power of neighbors
        neigh_pc = Pc[neighbors]

        # Get masker's power
        PM[k] = 10*np.log10(np.sum(np.power(10, 0.1*neigh_pc)))

    return PM

def STreduction(ST, c, Tq):

    # Get maskers' power
    PM = MaskPower(c, ST)

    # Get Tq thresholds for the maskers
    Tq_maskers = Tq[ST]

    # Get indices of all maskers that are above the treshold
    r_tq_idx = PM >= Tq_maskers

    # Get the maskers
    STr_tq = ST[r_tq_idx]
    PMr_tq = PM[r_tq_idx]
    # Now remove maskers based on the barks

    # Get freq of maskers
    freq = discrete2Hz(STr_tq)
    
    # Get barks of maskers
    barks = Hz2Barks(freq)

    # List of extra maskers to remove
    to_remove = []
    for i in range(len(STr_tq)):
        ki = STr_tq[i]

        bark_i = barks[i]
        for j in range(i, len(STr_tq)):
            bark_j = barks[j]

            # Compare barks, if difference less than 0.5, remove the ith masker (smaller)
            if bark_j - bark_i < 0.5:
                to_remove.append(ki)
                break

    # Remove them from the kept maskers
    idx = np.ravel([np.where(STr_tq == i) for i in to_remove])

    STr = STr_tq[idx]
    PMr = PMr_tq[idx]

    return STr, PMr

def SpreadFunc(ST, PM, Kmax):

    # Initialize the Sf matrix
    Sf = np.zeros((Kmax, len(ST)))

    # Iterate through each masker
    for j in range(len(ST)):

        # Get the masker's discrete freq
        k = ST[j]

        # Get its bark and power
        zk = Hz2Barks(discrete2Hz(k))
        PMk = PM[j]

        # For each discrete freq calculate the masker's contribution
        for i in range(Kmax):

            # Get discrete freq's bark
            zi = Hz2Barks(discrete2Hz(i))

            # Get difference
            Dz = zi - zk

            # Implement the spread function as described
            value = 0
            if -3 <= Dz < -1:
                value = 17*Dz + 0.4*PMk + 11
            elif -1 <= Dz < 0:
                value = (0.4*PMk + 6)*Dz
            elif 0<= Dz < 1:
                value = -17*Dz
            elif 1 <= Dz < 8:
                value = (0.15*PMk - 17)*Dz - 0.15*PMk

            # Set the value in the Sf matrix
            Sf[i,j] = value
    
    return Sf
        
def Masking_Thresholds(ST, PM, Kmax):
    
    # Initialize the Ti matrix
    Ti = np.zeros((Kmax, len(ST)))

    # Get Sf matrix
    Sf = SpreadFunc(ST, PM, Kmax)

    # Iterate through each masker
    for j in range(len(ST)):

        # Get the masker's discrete freq
        k = ST[j]

        # Get its bark and power
        zk = Hz2Barks(discrete2Hz(k))
        PMk = PM[j]

        # For each discrete freq calculate T_M (contribution in threshold)
        for i in range(Kmax):
            Ti[i,j] = PMk - 0.275*zk + Sf[i,j] - 6.025

    return Ti

# Calculate the global masking thresholds for reach discrete frequency
def Global_Masking_Thresholds(Ti, Tq):

    # Initialize an array to hold the thresholds
    Kmax = Ti.shape[0]
    Tg = np.zeros(Kmax)

    # For each frequency calculate the threshold
    for i in range(Kmax):

        # First term inside the log
        first_term = np.power(10, 0.1*Tq[i])

        # Second term inside the log
        second_term = np.sum(np.power(10, 0.1*Ti[i,:]))

        # Final calculation for the threshold of discrete freq i
        Tg[i] = 10*np.log10(first_term + second_term)

    return Tg

def psycho(c, D, Tq):

    Kmax = 1152
    
    ST, PM = STreduction(STinit(c, D), c, Tq)
    Ti = Masking_Thresholds(ST, PM, Kmax)
    Tg = Global_Masking_Thresholds(Ti, Tq)
    return Tg