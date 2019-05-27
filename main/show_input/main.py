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
parser.add_argument("--input",default="data.cur",help='Input file')
parser.add_argument("--xlabel",default="meV",help='Units')
args = parser.parse_args()


(V,dIdV) = dataset.openfile(args.input) # get the data

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)
currentfig = plt.gcf()
currentfig.canvas.set_window_title('Input dIdV')


# plot the tip DOS
plt.plot(V,dIdV,c="red",linewidth=5)
plt.xlabel("Energy ["+args.xlabel+"]")
plt.ylabel("dIdV [a.u.]")
plt.xlim([min(V),max(V)])
plt.ylim([0,max(dIdV)*1.5])
plt.show()
