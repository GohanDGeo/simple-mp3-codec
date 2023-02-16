# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# helper.py
# 
# This file contains helper functions for the project.
# Namely, a Hz2Barks which translates frequency to barks, discrete2Hz which translates discrete
# frequency to Hz.
# In addition, a function compression_ratio, returns the compression ratio between two files.

import numpy as np
import os

# Calculates barks from frequency
def Hz2Barks(f):
    z = 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)
    return z

# Calculate frequency in Hz from discrete frequency
def discrete2Hz(k):
    B = 689
    N = 36
    return k * B/N

# Calculate and return the compression ratio between 2 files (uncompressed, compressed),
# along with the compressed and uncompressed size in kilobytes
def compression_ratio(uncompressed, compressed):

    compressed_size = os.stat(compressed).st_size
    uncompressed_size = os.stat(uncompressed).st_size

    return uncompressed_size/1024, compressed_size/1024, uncompressed_size/compressed_size
    