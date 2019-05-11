#!/usr/bin/python
import os
path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(path+"/../../src") # add the library

import numpy as np
import sys
from stmdeconvpy import dataset

if len(sys.argv)>1:
    args = sys.argv[1:] # get the arguments
else: args = ["--input","2d.data"] # default workaround


from copy import deepcopy
argsin = deepcopy(args) # copy

for i in range(len(args)):
    if args[i]=="--help": 
      os.system("stmdeconvpy --help")
      exit()
        
for i in range(len(args)):
    if args[i]=="--input": 
        name = args[i+1]
        del argsin[i+1]
        del argsin[i]

instr = ""
for s in argsin: instr += " "+s

name = "2d.data"
m = dataset.mapsplit(name) # read the data
# create output files
x = []
y = []
z = []
d = []
os.system("mkdir stmdeconvtmp")
print("Creating folder stmdeconvtmp, it will be cleaned at the end")
os.chdir("stmdeconvtmp") # go to the folder
for i in range(len(m)):
    mi = m[i]
    np.savetxt("temp.txt",np.array([mi[0],mi[2]]).T)
    os.system("stmdeconvpy --show false --input temp.txt "+instr)
    out = np.genfromtxt("DECONVOLUTED_DOS.OUT").T # get the data
    x = np.concatenate([x,out[0]])
    y = np.concatenate([y,mi[1]])
    z = np.concatenate([z,out[1]])
    d = np.concatenate([d,out[2]])
    np.savetxt("../DECONVOLUTED_DOS_MAP.OUT",np.array([x,y,z,d]).T)
    print("Saved data in DECONVOLUTED_DOS_MAP.OUT")
os.system("cp TIP_DOS.OUT ../") # copy to the previous folder
os.chdir("..")
os.system("rm -rf stmdeconvtmp") # cleaning temporal folder
print("Deconvolution finished")

