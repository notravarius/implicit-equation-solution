import numpy as np
from numpy import arange
from scipy.optimize import fsolve

def f(z,k):

    """Function to minimize with solver
    Args: 
        k: constant 
        z: starting value"""

    return z - np.log(k*z + 1) 

k_values = []
z_values = []

for k in arange(1,5,0.0001): # since K is always less than 5
    z = fsolve(f,100,k) # get z by minimizing f
    k_values.append(k)
    z_values.append(z[0])

# creating a 4th degree polynomial logarithm regression model by minimizing  
# the loss function given the datapoints obtained before.

log_reg = np.polyfit(np.log(k_values), z_values, 4) 

import matplotlib.pyplot as plt

# plot both actual data point and regression model

k = np.linspace(1, 5, 100)

y = (log_reg[0]*np.log(k)**4 + 
    log_reg[1]*np.log(k)**3 + 
    log_reg[2]*np.log(k)**2 + 
    log_reg[3]*np.log(k) + 
    log_reg[4])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.plot(k, y, "r")
plt.plot(k_values, z_values)
plt.xlabel('K', size=15)
plt.ylabel('Z', size=15)
plt.title('Solution Space for Z = ln(KZ + 1)', size=20)
plt.grid(True)
plt.legend(["4th degree logarithm regression", "Z = ln(KZ + 1)"], loc ="lower right", prop={'size': 15})

plt.show()