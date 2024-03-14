#!/usr/bin/env python3

from first_itds_modules import entropy
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0001, 0.9999, 1000)  # set of possible probabilities (x axis)
y=[] # set of entropies (y axis)
for i in range(0, len(x)): # calculation of each axis
    pair=[x[i],1-x[i]] # probability of each value = 1 - probability of other value
    #print(pair)
    y.append(entropy(pair)) # calculate entropy & add it to results
#print (y)
plt.plot(x, y, "b-", label="H(X)")
plt.xlabel("P0")
plt.ylabel("H(X)")
plt.title("Entropy of Discrete R.V.")
plt.legend()
plt.show()