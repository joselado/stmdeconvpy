#!/usr/bin/python
import os
path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(path+"/../../src") # add the library

import numpy as np
from stmdeconvpy.singledeconvolution import single_deconvolution,write_single



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default="data.cur",help='Input file with the data')
    parser.add_argument("--output",default="DECONVOLUTED_DOS.OUT",help='Output file with the deconvolved DOS')
    parser.add_argument("--tip_output",default="TIP_DOS.OUT",help='Output file with the DOS of the tip (including temperature)')
    parser.add_argument("--dIdV_output",default="dIdV_OUTPUT.OUT",help='Output file with the resulting dIdV obtained assuming the deconvoluted DOS')
    parser.add_argument("--dIdV_input",default="dIdV_INPUT.OUT",help='Input file with the dIdV after it was preprocessed')
    parser.add_argument("--delta",default=0.12,help='Superconducting gap of the tip')
    parser.add_argument("--Ttip",default=0.02,help='Temperature of the tip')
    parser.add_argument("--Tsur",default=0.02,help='Temperature of the surface')
    parser.add_argument("--Tdos",default=0.02,help='Temperature of the surface DOS')
    parser.add_argument("--ntries",default=4,help='Take the best out of ntries minimizations')
    parser.add_argument("--maxn",default=100,help='Maximum number of grid points used in the deconvolution, increase it for higher accuracy')
    parser.add_argument("--gamma",default=0.01,help='Gamma smearing of the Dynes superconducting DOS')
    parser.add_argument("--show",default="true",help='Show the result')
    parser.add_argument("--mode",default="minimize",help='Mode of the calculation, minimize is the feault, whereas algebra is faster')
    args = parser.parse_args()

    out = single_deconvolution(args) # call the function      
    write_single(args,out) # write the data
    
    if args.show!="true": exit() # return
    
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

