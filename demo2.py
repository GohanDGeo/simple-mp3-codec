import numpy as np

test = []

for i in range(5):
    test.append(i*np.ones(3))

test = np.asarray(test).ravel()

print(test)