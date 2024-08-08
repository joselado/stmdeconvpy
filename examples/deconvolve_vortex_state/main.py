import os
import sys
sys.path.append("../../src")



import numpy as np



from stmdeconvpy import deconvolve
from stmdeconvpy import profiles

xs = np.linspace(-4.0,4.0,1000)

# define filtering signal
yf = profiles.dynes_superconductor(delta=0.5,gamma=0.1)(xs)

yf = profiles.step_tip(delta=1.0,lowest=0.1,highest=1.0)(xs)
yf = yf/np.mean(yf)


# define the real signal
y0 = profiles.vortex_state(delta=0.3)(xs)
y0 = y0/np.mean(y0)
#y0 +=  0.1*np.random.random(len(y0))

# compute the dIdV associated to that DOS
(xc,yc) = deconvolve.dos2dIdV(xs,y0,xs,yf)

yc = yc/np.mean(yc)


import matplotlib.pyplot as plt

# deconvolve the signal, from dIdV 2 DOS
xn,ydc = deconvolve.dIdV2dos(xs,yc,xs,yf,n=200,mode="minimize")
#ydc = ydc/np.max(ydc)

# and do it again to compare
(xc2,yc2) = deconvolve.dos2dIdV(xs,ydc,xs,yf)
yc2 = yc2/np.mean(yc2)



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
plt.plot(xc,yc,label="Measured dI/dV (real)",c="green")
plt.plot(xc2,yc2,label="Measured dI/dV (inferred)",c="blue")
plt.xlim([np.min(xs),np.max(xs)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")
plt.legend()


plt.subplot(1,3,3)
plt.plot(xs,y0,label="Real DOS",c="red")
plt.plot(xn,ydc,label="Deconvolved DOS",c="blue")
plt.xlim([np.min(xs),np.max(xs)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV or DOS [a.u.]")
plt.legend()

plt.tight_layout()
plt.show()

