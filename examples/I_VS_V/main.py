import os
import sys
sys.path.append("../../src")



import numpy as np



from stmdeconvpy import deconvolve
from stmdeconvpy import profiles

xs = np.linspace(-2.0,2.0,30)

# define the superconducting DOS
ytip = profiles.superconducting(delta=0.5)(xs)
ytip = profiles.constant()(xs)

# define the real DOS
ysur = profiles.random_peaks(nmax=3,xmin=-1.0,xmax=1.0,wmin=0.05,wmax=0.2)(xs)
#ysur = profiles.peak()(xs) # return a single peak


# define the convolved signal
#(xs,yexp) = deconvolve.dos2dIdV(xs,ysur,xs,ytip)
(xs,yexp) = deconvolve.dos2I(xs,ysur,xs,ytip)


import matplotlib.pyplot as plt

# deconvolve the signal
#xn,ydc = deconvolve.deconvolve_dIdV(xs,yexp,xs,ytip,sol=ysur)
#xn,ydc = deconvolve.deconvolve_dIdV(xs,yexp,xs,ytip,sol=None)
xn,ydc = deconvolve.deconvolve_I(xs,yexp,xs,ytip,sol=None)
(xs,yexp2) = deconvolve.dos2I(xs,ydc,xs,ytip)
#xn,ydc = deconvolve.deconvolve(xs,yexp,xs,ytip,sol=ysur)


# now plot the results
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)



plt.plot(xs,ytip,label="Tip DOS",c="blue")
plt.plot(xs,yexp,label="Measured (convolution)",c="green")
plt.scatter(xs,yexp2,label="Convolution of the solution",c="green")
plt.plot(xs,ysur,label="Surface DOS",c="red")
plt.scatter(xn,ydc,label="Deconvolved surface DOS",c="red")
plt.xlim([np.min(xs),np.max(xs)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")
plt.legend()
plt.show()

