#!/usr/bin/python
import os
path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(path+"/../src") # add the library

import numpy as np
from stmdeconvpy import deconvolve
from stmdeconvpy import profiles
from stmdeconvpy import dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input",default="data.cur",help='Input file with the data')
parser.add_argument("--output",default="DECONVOLUTED_DOS.OUT",help='Output file with the deconvolved DOS')
parser.add_argument("--tip_output",default="TIP_DOS.OUT",help='Output file with the DOS of the tip (including temperature)')
parser.add_argument("--dIdV_output",default="dIdV_OUTPUT.OUT",help='Output file with the resulting dIdV obtained assuming the deconvoluted DOS')
parser.add_argument("--dIdV_input",default="dIdV_INPUT.OUT",help='Input file with the dIdV sfter it was preprocessed')
parser.add_argument("--delta",default=0.12,help='Superconducting gap of the tip')
parser.add_argument("--T",default=0.02,help='Temperature of the tip')
parser.add_argument("--ntries",default=4,help='Take the best out of ntries minimizations')
args = parser.parse_args()

print("Reading data from ",args.input)
print("The script will perform ",args.ntries,"minimizations")

# get the data
(V,dIdV_exp) = dataset.opencur(args.input) # get the data
dIdV_exp = profiles.normalize(dIdV_exp)





I_exp = deconvolve.dIdV2I(V,dIdV_exp) # get the current

#V,I_exp = deconvolve.expand_I(V,I_exp)
V,dIdV_exp = deconvolve.I2dIdV(V,I_exp)



print("Processed input data written to ",args.dIdV_input)
np.savetxt(args.dIdV_input,np.array([V,dIdV_exp]).T)



T = float(args.T) # get the temperature
delta = float(args.delta) # get the superconducting gap

# define the superconducting DOS
dos_tip = profiles.superconducting(delta=delta,T=T)(V) # superconducting Tip


print("Tip DOS written to ",args.tip_output)
np.savetxt(args.tip_output,np.array([V,dos_tip]).T)

#dos_tip = profiles.constant()(V) # normal Tip




#####################################################################
############# You do not need to change anything else ###############
#####################################################################



import matplotlib.pyplot as plt

# deconvolve the signal
xn,dos_sur_dc = deconvolve.deconvolve_I(V,I_exp,V,dos_tip,sol=None,
        ntries=int(args.ntries),print_error=False,T=T)

# write the surface DOS
np.savetxt(args.output,np.array([xn,dos_sur_dc]).T)
print("Surface DOS written to ",args.output)

(V,I_exp2) = deconvolve.dos2I(V,dos_sur_dc,V,dos_tip,T=T)
(V,dIdV_exp2) = deconvolve.dos2dIdV(V,dos_sur_dc,V,dos_tip,T=T)
print("dIdV with deconvoluted DOS written to ",args.dIdV_output)
np.savetxt(args.dIdV_output,np.array([V,dIdV_exp2]).T)
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
plt.xlim([np.min(V),np.max(V)])
plt.xlabel("Energy [a.u.]")
plt.ylabel("DOS [a.u.]")
plt.legend()
plt.show()

