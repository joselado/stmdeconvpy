import os
import sys
sys.path.append("../../src")
import numpy as np
from stmdeconvpy import deconvolve
from stmdeconvpy import profiles
from stmdeconvpy import dataset


# get the data
(V,dIdV_exp) = dataset.opencur("data.cur") # get the data
dIdV_exp = profiles.normalize(dIdV_exp)
I_exp = deconvolve.dIdV2I(V,dIdV_exp) # get the current

#V,I_exp = deconvolve.expand_I(V,I_exp)
V,dIdV_exp = deconvolve.I2dIdV(V,I_exp)

x,out = deconvolve.convolve_single_dfd(V,dIdV_exp,T=0.04)
#print(out.shape,V.shape) ; exit()

# define the superconducting DOS
dos_tip = profiles.superconducting(delta=0.12,T=0.02)(V) # superconducting Tip
#np.savetxt("DOS.OUT",np.array([V,dos_tip]).T) ;  exit()
#dos_tip = profiles.constant()(V) # normal Tip




#####################################################################
############# You do not need to change anything else ###############
#####################################################################



import matplotlib.pyplot as plt

# deconvolve the signal
xn,dos_sur_dc = deconvolve.deconvolve_I(V,I_exp,V,dos_tip,mode="algebra",
        ntries=1)
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
#plt.plot(V,dos_sur,label="Surface DOS",c="red")
plt.scatter(xn,dos_sur_dc,label="Deconvolved surface DOS",c="blue")
xn,dos_sur_dc2 = deconvolve.convolve_single_dfd(xn,dos_sur_dc,T=0.01)
plt.scatter(xn,dos_sur_dc2,label="DOS finite T",c="red")
plt.xlim([np.min(V),np.max(V)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("DOS [a.u.]")
plt.legend()
plt.show()

