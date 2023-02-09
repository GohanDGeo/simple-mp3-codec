import numpy as np
from helper import Hz2Barks, discrete2Hz
from dct import *

# Calculate the neighborhood of each discrete freq k
def Dksparse(Kmax):
    D = np.zeros((Kmax, Kmax))
    for k in range(Kmax):
        idx = []
        if 2 < k < 282:
            idx = [k-2, k+2]
        elif 282 <= k < 570:
            n_range = np.arange(2, 14)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
        elif 570 <= k < 1152:
            n_range = np.arange(2, 28)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
            idx = idx[idx < Kmax]
        
        if len(idx) > 0:
            D[k, idx] = 1
    
    return D

# Get tonal components
def STinit(c, D):
    # Get power of coefficients
    Pc = DCTpower(c)

    # Initialize ST list
    ST = []

    # Check if it is a tonal component
    for k in range(len(c)):
        # Compare the power of the kth coefficient with each left and right coeff. and
        # with its Dk neighbors
        neighbors = np.array([k-1, k+1])

        # If k == 1151, it does not have a right neighbor, so only check the left and Dk
        neighbors = neighbors[neighbors < len(c)]
        neighbors = neighbors[neighbors >= 0]
        
        # And compare.

        #Dk = Dneighbors[k]
        Dk = np.nonzero(D[k,:])
        if np.all(Pc[k] > Pc[neighbors]) or np.all(Pc[k] > (Pc[Dk] + 7)):
            ST.append(k)

    ST = np.asarray(ST)

    return ST

# Calculate the power of the maskers
def MaskPower(c, ST):
    Pc = DCTpower(c)
    PM = np.zeros(len(ST))

    # For each masker
    for k in range(len(ST)):

        # Get the masker's neighbors and itself
        neighbors = np.array([k-1, k, k+1])

        # Check if the neighbors exist (within bounds)
        neighbors = neighbors[neighbors < len(Pc)]
        neighbors = neighbors[neighbors >= 0]

        # Get power of neighbors and of the masker
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
    for i in range(len(STr_tq)-1):
        bark_i = barks[i]
        bark_j = barks[i+1]

        # Compare barks, if difference less than 0.5, remove the ith masker (smaller)
        if bark_j - bark_i < 0.5:
            to_remove.append(i)

    # Remove them from the kept maskers
    STr = np.delete(STr_tq, to_remove)
    PMr = np.delete(PMr_tq, to_remove)

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
                value = 17*Dz - 0.4*PMk + 11
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