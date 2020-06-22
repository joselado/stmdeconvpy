import os
import sys
sys.path.append("../../src")
import numpy as np
from stmdeconvpy import deconvolve
from stmdeconvpy import profiles
import matplotlib.pyplot as plt
V = np.linspace(-4.0,4.0,200) # energies

# define the superconducting DOS
dos_tip = profiles.superconducting(delta=0.5,T=0.1)(V) # superconducting Tip
#dos_tip = profiles.constant()(V) # normal Tip
np.savetxt("TIP.OUT",np.array([V,dos_tip]).T)# ; exit()



# define the real DOS
dos_sur = profiles.random_peaks(nmin=3,nmax=3,xmin=-1.0,xmax=1.0,wmin=0.05,wmax=0.2)(V)
#dos_sur = profiles.peak(w=0.1)(V)
# and its associated signals (for testing purpose)
(V,I_exp) = deconvolve.dos2I(V,dos_sur,V,dos_tip)
# add some noise to the signal
#I_exp += np.random.random(len(I_exp))*0.2*np.max(np.abs(I_exp))

# and compute the experimental dIdV
(V,dIdV_exp) = deconvolve.I2dIdV(V,I_exp)

#####################################################################
############# You do not need to change anything else ###############
#####################################################################







# deconvolve the signal
xn,dos_sur_dc = deconvolve.deconvolve_I(V,I_exp,V,dos_tip,
        mode="algebra")
(V,I_exp2) = deconvolve.dos2I(V,dos_sur_dc,V,dos_tip)
(V,dIdV_exp2) = deconvolve.dos2dIdV(V,dos_sur_dc,V,dos_tip)
#xn,dos_sur_dc = deconvolve.deconvolve(V,I_exp,V,dos_tip,sol=dos_sur)


# now plot the results
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)


# plot the measured I VS V
plt.subplot(223)
plt.plot(V,I_exp*100,label="Measured I",c="red")
plt.scatter(V,I_exp2*100,label="I of deconvoluted DOS",c="blue")
plt.legend()
plt.xlabel("Energy [a.u.]")
plt.ylabel("I [a.u.]")




# plot the measured dIdV VS V
plt.subplot(224)
plt.plot(V,dIdV_exp*100,label="Measured dIdV",c="red")
plt.scatter(V,dIdV_exp2*100,label="dIdV of deconvoluted DOS",c="blue")
plt.legend()
plt.xlabel("Energy [a.u.]")
plt.ylabel("dI/dV [a.u.]")






# plot the tip DOS
plt.subplot(221)
plt.plot(V,dos_tip,label="Tip DOS",c="black")
plt.xlabel("Energy [a.u.]")
plt.ylabel("DOS [a.u.]")
plt.legend()


# plot the DOS of the surface
plt.subplot(222)
plt.plot(V,dos_sur,label="Surface DOS",c="red")
plt.scatter(xn,dos_sur_dc,label="Deconvolved surface DOS",c="blue")
plt.xlim([np.min(V),np.max(V)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("DOS [a.u.]")
plt.legend()
plt.show()

