# MULTIMEDIA SYSTEMS, ECE AUTH 2022-2023
# KOUTROUMPIS GEORGIOS, 9668
# KYRGIAFINI-AGGELI DIMITRA, 9685
# 
# demo_quantization.py
# 
# A graphical demonstration of the quantizer/dequantizer used

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Set number of bits
b = 3

# Find decision boundaries, like in the quantizer
d = np.linspace(-1, 1, (2**b)+1)
d = np.delete(d, len(d)//2)

# Find middle point for each zone
mid_points = np.array([(d[i+1] + d[i])/2 for i in range(len(d)-1)])

# Plot the quantizer-dequantizer plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks(d)
ax.set_yticks(mid_points)
plt.step(d, np.insert(mid_points,0,mid_points[0]), color="c", linewidth="2")

plt.title(f"Quantizer-Dequantizer for b={b}")
plt.show()
