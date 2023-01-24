import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
sys.path.insert(1, 'scripts and data') 
import mp3
from subband_filtering import codec0

Tq = np.load("scripts and data//Tq.npy", allow_pickle=True)[0]
print(Tq.shape)