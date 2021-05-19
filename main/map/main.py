#!/usr/bin/python
import os
path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(path+"/../../src") # add the library

import numpy as np
import sys
from stmdeconvpy import dataset
from stmdeconvpy.stmpath import binpath
from stmdeconvpy.stmpath import binexecute
from stmdeconvpy import singledeconvolution

if len(sys.argv)>1:
    args = sys.argv[1:] # get the arguments
else: args = ["--input","2d.data"] # default workaround


from copy import deepcopy
argsin = deepcopy(args) # copy

for i in range(len(args)):
    if args[i]=="--help": 
      os.system("stmdeconvpy --help")
      exit()


def pread(name):
    """Read the value of a certain parameter"""
    for i in range(len(args)):
        if args[i]=="--"+name: return int(args[i+1])
    return None


for i in range(len(args)):
    if args[i]=="--input": 
        name = args[i+1]
        del argsin[i+1]
        del argsin[i]

instr = ""
for s in argsin: instr += " "+s

#name = "2d.data"
ny = pread("maxn") # maximum number of curves
print("Enforcing",ny,"curves")
m = dataset.mapsplit(name,ny=ny) # read the data
# create output variables
x = []
y = []
z = []
d = []
#
x2 = []
y2 = []
z2 = []
#os.system("mkdir stmdeconvtmp")
#print("Creating folder stmdeconvtmp, it will be cleaned at the end")
#os.chdir("stmdeconvtmp") # go to the folder
ms = [mi for mi in m] # loop over inputs
def compute_single(mi):
    print()
    print("Computing ",mi[1][0])
    print()
    inputdata = [mi[0],np.abs(mi[2])] # input data
    ins = singledeconvolution.string2dict(instr) # inputs
    ins["input"] = inputdata # inputdata
    out = singledeconvolution.single_deconvolution(ins) # compute
    return out


from stmdeconvpy import parallel
parallel.cores = parallel.maxcpu
outs = parallel.pcall(compute_single,ms) # compute all

for i in range(len(m)):
    mi = m[i]
#    out = compute_single(mi)
    out = outs[i]
#    singledeconvolution.write_single(ins,out) # write result
#    np.savetxt("temp.txt",np.array([mi[0],np.abs(mi[2])]).T)
#    binexecute("stmdeconvpy --show false --input temp.txt "+instr)
    # read the deconvoluted DOS
#    out = np.genfromtxt("DECONVOLUTED_DOS.OUT").T # get the data
    x = np.concatenate([x,out["V_dos"]])
    y = np.concatenate([y,mi[1]])
    z = np.concatenate([z,out["dos"]])
#    if len(out)==3:  d = np.concatenate([d,out[2]])
    d = np.concatenate([d,mi[1]*0.])
#    print("Saved data in DECONVOLUTED_DOS_MAP.OUT")
    # now read the recovoluted dIdV
#    out2 = np.genfromtxt("dIdV_OUTPUT.OUT").T # get the data
    x2 = np.concatenate([x2,out["V_exp2"]])
    y2 = np.concatenate([y2,mi[1]])
    z2 = np.concatenate([z2,out["dIdV_exp2"]])
#    np.savetxt("../dIdV_OUTPUT_MAP.OUT",np.array([x2,y2,z2]).T)
#    np.savetxt("../TIP_DOS.OUT",np.array([out["V_tip"],out["dos_tip"]]).T)

# write all the data
np.savetxt("dIdV_OUTPUT_MAP.OUT",np.array([x2,y2,z2]).T)
np.savetxt("TIP_DOS.OUT",np.array([out["V_tip"],out["dos_tip"]]).T)
np.savetxt("DECONVOLUTED_DOS_MAP.OUT",np.array([x,y,z,d]).T)
#os.chdir("..")
#os.system("rm -rf stmdeconvtmp") # cleaning temporal folder
print("Deconvolution finished")


#os.system("stmdeconvpy_plotmap") # plot results
