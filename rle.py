# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# rle.py
#
# This file contains the functions for run-length encoding.

# Imports
import numpy as np

# Given a series of symbols @symb_index, perform run-length encoding
def RLE(symb_index, K):

    # Initialize the count 
    count = 0

    # Set the first previous symbol as the first in the array
    #s_old = symb_index[0]

    # A list holding the tuples (quant_symbol, following_zeros)
    run_symbols = []

    # For each symbol in the symbol indices
    for i in range(len(symb_index)):

        # Get the new symbol
        s = symb_index[i]

        # Check if it is a following zero (and not the last element)
        # (If it is the last element, a symbol in the form of (0, N) will be added, where N the N last 0s)
        if s == 0:
            count += 1
        # Else, append the symbol and its length as a tuple and reset the count
        else:
            run_symbols.append((s, count))
            count = 0

    # This means that if the last N symbols of @symb_indx are 0s, they won't be included in the encoding.
    # This is OK, since during decoding the run-length encoding, the information of the lenght of the original
    # array is known. So the left over symbols are set to 0.
    # Eg. [0 0 0 1 2 0 0 0 0 0 3 0 0 0 0] will be encoded as [(3,1) (0,2) (5,3)], ignoring the last four 0s
    # as they can be retrieved knowing the run symbols and the length of the original array of symbols.

    # If however @symb_index consists only of 0s, one symbol will be produced, in the form of (0, @(K-1))
    if len(run_symbols) == 0:
        run_symbols = [(0, K - 1)]
    return run_symbols

# Given the run-length encoding symbols, and the length of the original symbol array, 
# perform the inverse of the run-length encoding
def IRLE(run_symbols, K):

    # Initialize an array of symbols to 0s.
    # This means that any 0s at the end of the original symbol array that have not been encoded,
    # are set to 0 at this step.
    symb_index = np.zeros(K)

    # Initialize an index showing the index to currently add to in the symb_index array
    idx = 0

    # For each pair of (symbol, length)
    for symbol_pair in run_symbols:

        # Get the symbol and its run length
        symbol = symbol_pair[0]
        following_zeros = symbol_pair[1]

        # Add the current symbol at the appropriate place.
        # The array is already populated with zeros, so only the symbol
        # in the tuple pair needs to be added. Its position is 
        # the position of the index plus the number of following zeros
        # Eg. @idx = 0 and @symbol = 2 and @following_zeros = 3
        # meaning in the original symbol array the sequence 0002 was encoded into
        # (2,3). Thus in the position @idx + @following_zeros = 3, the symbol 2 will
        # be inserted ([0 0 0 2 ...])
        symb_index[idx+following_zeros] = symbol

        # Update the current index
        idx += following_zeros + 1

    return np.int16(symb_index)