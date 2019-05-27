#!/usr/bin/python
import os
path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(path+"/../../src") # add the library

import numpy as np
from stmdeconvpy import deconvolve
from stmdeconvpy import profiles
from stmdeconvpy import dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tip_output",default="TIP_DOS.OUT",help='Output file with the DOS of the tip (including temperature)')
parser.add_argument("--delta",default=0.12,help='Superconducting gap of the tip')
parser.add_argument("--Ttip",default=0.02,help='Temperature of the tip')
parser.add_argument("--Vwindow",default=10,help='Energy window')
parser.add_argument("--gamma",default=0.01,help='Gamma smearing of the Dynes superconducting DOS')
parser.add_argument("--show",default="true",help='Show the result')
args = parser.parse_args()


V = np.linspace(-args.Vwindow,args.Vwindow,1000)

delta = float(args.delta) # get the superconducting gap
gamma = float(args.gamma) # get the superconducting gamma

# define the superconducting DOS
dos_tip = profiles.dynes_superconductor(delta=delta,gamma=gamma)(V) # superconducting Tip
print("Tip DOS written to ",args.tip_output)
np.savetxt(args.tip_output,np.array([V,dos_tip]).T)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)
currentfig = plt.gcf()
currentfig.canvas.set_window_title('Tip DOS')


# plot the tip DOS
plt.plot(V,dos_tip,label="Tip DOS",c="red",linewidth=5)
plt.xlabel("Energy [meV]")
plt.ylabel("DOS [a.u.]")
plt.xlim([min(V),max(V)])
plt.ylim([0,max(dos_tip)*1.5])
plt.legend()
plt.show()
