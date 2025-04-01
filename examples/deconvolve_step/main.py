import os
import sys
sys.path.append("../../src")



import numpy as np



from stmdeconvpy import deconvolve
from stmdeconvpy import profiles

# signal has many peaks, allow more kinetic energy than usual
deconvolve.kinetic_quench = 1e-3

xs = np.linspace(-4.0,4.0,200)

# define filtering signal
yf = profiles.dynes_superconductor(delta=0.4,gamma=0.05)(xs)

# define the real signal, just a couple of steps
y0 = profiles.step_tip(delta=2.5,lowest=0.3,highest=0.8)(xs)
y0 += profiles.step_tip(delta=1.0,lowest=0.5,highest=1.2)(xs)
y0 += profiles.step_tip(delta=1.5,lowest=0.5,highest=1.2)(xs)
y0 = y0/np.max(y0)


# compute the dIdV associated to that DOS
(xc,yc) = deconvolve.dos2dIdV(xs,y0,xs,yf)

yc = yc/np.max(yc)


import matplotlib.pyplot as plt

# deconvolve the signal, from dIdV 2 DOS
deconvolve.kinetic_quench = 1e-7 # allow for strong steps
ntries = 3 # several tries
xn,ydc = deconvolve.dIdV2dos(xs,yc,xs,yf,n=200,ntries=ntries,mode="minimize")

ydc = ydc/np.max(ydc)

(xc2,yc2) = deconvolve.dos2dIdV(xs,ydc,xs,yf)
yc2 = yc2/np.max(yc2)


# now plot the results
import matplotlib

fig = plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Tip DOS")
plt.plot(xs,yf,label="Tip DOS",c="blue")
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")



plt.subplot(1,3,2)
plt.title("Measured dI/dV")
plt.plot(xs,yc,label="Real dIdV",c="green")
plt.plot(xs,yc2,label="dIdV from deconvoluted DOS",c="blue")
plt.xlim([np.min(xs),np.max(xs)])
plt.legend()
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")


plt.subplot(1,3,3)
plt.plot(xs,y0,label="Real DOS",c="red")
plt.plot(xn,ydc,label="Deconvolved DOS",c="blue")
plt.xlim([np.min(xs),np.max(xs)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")
plt.legend()

plt.tight_layout()
plt.show()

