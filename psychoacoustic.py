# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# psychoacoustic.py
#
# This file contains the functions that implement the psychoacoustic model.

# Imports
import numpy as np
from helper import Hz2Barks, discrete2Hz
from dct import *

# Calculate the neighborhood of each discrete freq k.
# Each row corresponds to a discrete frequency k.
# Every row is initialized at 0, but according to the formula given in the project,
# for every other dicrete freq j that is in th neighberhood Dk of k, 
# The element D[k, j] is set to 1
def Dksparse(Kmax):
    # Initialize the matrix D
    D = np.zeros((Kmax, Kmax))
    
    # Then find the neighborhood for each k
    for k in range(Kmax):
        # A list containing the indices of k's neighbors
        idx = []

        # Check in which range k is in, 
        # and determine its neighbors accordingly
        if 2 < k < 282:
            # Here the neighbors are the neighbor 2 to the left
            # and 2 to the right
            idx = [k-2, k+2]
        elif 282 <= k < 570:
            # First create a list from 2 to 14 ([2, 3, 4, ...,  14])
            n_range = np.arange(2, 14)
            # Then the neighbors are [k-14, k-13, k-12, ... , k-2, k+2, k+3, ... , k+14]
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
        elif 570 <= k < 1152:
            # Same here
            n_range = np.arange(2, 28)
            idx = np.concatenate(((-1)*np.flip(n_range), n_range), axis=None) + k
            # And check that the neighbor indices do not go out of bounds (for the last discrete freqs)
            idx = idx[idx < Kmax]
        
        # Then set the neighbors to 1
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
        # Get the indices of the left and right neihbor
        neighbors = np.array([k-1, k+1])

        # Check that they are within bounds
        neighbors = neighbors[neighbors < len(c)]
        neighbors = neighbors[neighbors >= 0]
        
        # Get the other neighbors, according to Dk
        Dk = np.nonzero(D[k,:])

        # And check if either condition is true
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

        # Get the masker's left-right neighbors and itself
        neighbors = np.array([k-1, k, k+1])

        # Check if the neighbors exist (within bounds)
        neighbors = neighbors[neighbors < len(Pc)]
        neighbors = neighbors[neighbors >= 0]

        # Get power of neighbors and of the masker
        neigh_pc = Pc[neighbors]

        # Get masker's power
        PM[k] = 10*np.log10(np.sum(np.power(10, 0.1*neigh_pc)))

    return PM

# Reduces the tonal components
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

# Function that defines the spread function
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

# Function that returns the masking thresholds
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

# Function that performs the psychoacoustic model from start to finish
def psycho(c, D, Tq):

    # Set the number of discrete frequencies
    Kmax = 1152
    
    # Get the tonal components and their power
    ST, PM = STreduction(STinit(c, D), c, Tq)
    
    # Get the masking thresholds
    Ti = Masking_Thresholds(ST, PM, Kmax)

    # And the global masking thresholds
    Tg = Global_Masking_Thresholds(Ti, Tq)

    # And return the global masking thresholds
    return Tg