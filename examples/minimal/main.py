import os
import sys
sys.path.append("../../src")



import numpy as np



from stmdeconvpy import deconvolve
from stmdeconvpy import profiles

xs = np.linspace(-2.0,2.0,200)

# define filtering signal
yf = profiles.superconducting(delta=0.5)(xs)

# define the real signal
y0 = profiles.random_peaks(nmax=6,xmin=-2.0,xmax=2.0,wmin=0.05,wmax=0.2)(xs)


# define the convolved signal
yc = deconvolve.fdconvolution(xs,y0,yf)


import matplotlib.pyplot as plt

# deconvolve the signal
xn,ydc = deconvolve.deconvolve(xs,yc,xs,yf)


# now plot the results
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)



plt.plot(xs,yf,label="Tip DOS",c="blue")
plt.plot(xs,yc,label="Measured dI/dV (convolution)",c="green")
plt.plot(xs,y0,label="Surface DOS",c="red")
plt.scatter(xn,ydc,label="Deconvolved surface DOS",c="red")
plt.xlim([np.min(xs),np.max(xs)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")
plt.legend()
plt.show()

