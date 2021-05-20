import numpy as np
from . import deconvolve
from . import profiles
from . import dataset


def dict2obj(args):
    if type(args)==dict: # transform to object
        from collections import namedtuple
        args = namedtuple("Params", args.keys())(*args.values())
    return args


def single_deconvolution(args):
    """Perform a single deconvolution"""
    args = dict2obj(args) # convert to object
    out = dict() # dictionary
    #### preprocessing of the arguments
    ntries = int(args.ntries)
    if args.mode=="algebra":
        print("Calculation mode is algebra, taking a single try")
        ntries = 1
    
#    print("The script will perform ",args.ntries,"minimizations")
    
    # get the data
    if type(args.input)==str:
        print("Reading data from ",args.input)
        (V0,dIdV_exp0) = dataset.openfile(args.input) # get the data
    else:
        V0,dIdV_exp0 = args.input[0],args.input[1] # get the data
    Vmin,Vmax = np.min(V0),np.max(V0) # minimum and maximum
    fcrop = dataset.crop(Vmin,Vmax) # function to crop the data
    (V,dIdV_exp) = dataset.symmetric_bounds(V0,dIdV_exp0) # use symmetric window
    
    
    I_exp = deconvolve.dIdV2I(V,dIdV_exp) # get the current
    
    #V,I_exp = deconvolve.expand_I(V,I_exp)
    V,dIdV_exp = deconvolve.I2dIdV(V,I_exp)
    
    
    
#    print("Processed input data written to ",args.dIdV_input)
    out["V_exp"] = V.copy()
    out["dIdV_exp"] = dIdV_exp.copy()
    
    
    
    delta = float(args.delta) # get the superconducting gap
    gamma = float(args.gamma) # get the superconducting gamma
    
    # define the superconducting DOS
    dos_tip = profiles.dynes_superconductor(delta=delta,gamma=gamma)(V) # superconducting Tip
    
    
#    print("Tip DOS written to ",args.tip_output)
    out["V_tip"] = V.copy()
    out["dos_tip"] = dos_tip.copy()
    
    
    #####################################################################
    ############# You do not need to change anything else ###############
    #####################################################################
    
    maxn = 2*int(args.maxn) # points in the deconvolution
    
    
    Ttip = float(args.Ttip) # temperature of the tip
    Tsur = float(args.Tsur) # temperature of the tip
    
    
    # add the temperature smearing to the tip
    def f(x,y):
      (x,y) = deconvolve.dos2dIdV(x,y,x,y*0.0+1.0,Ttip=Ttip,Tsur=Ttip)
      out["V_tipT"] = x.copy()
      out["dos_tipT"] = y.copy()
    f(V,dos_tip) # write in file
    
    
    
    
    # deconvolve the signal
    xn,dos_sur_dc,error = deconvolve.deconvolve_I(V,I_exp,V,dos_tip,
            return_error = True,n=maxn,
            ntries=int(ntries),print_error=False,Ttip=Ttip,
            Tsur=Tsur,mode=args.mode)
    
    
    
    # write the surface DOS
    def f(x,y,T):
      if T>0.0: (x,y) = deconvolve.convolve_single_dfd(x,y,T=T)
      [x,y] = fcrop(x,y)
      out["V_dos"] = x.copy()
      out["dos"] = y.copy()
    Tdos = float(args.Tdos)
    f(xn,dos_sur_dc,T=Tdos) # write in file
    
    
#    print("Surface DOS written to ",args.output)
    
    (V,I_exp2) = deconvolve.dos2I(V,dos_sur_dc,V,dos_tip,
            Ttip=Ttip,Tsur=Tsur)
    (V,dIdV_exp2) = deconvolve.dos2dIdV(V,dos_sur_dc,V,dos_tip,
            Ttip=Ttip,Tsur=Tsur)
    [V,dIdV_exp2] = fcrop(V,dIdV_exp2)
    out["V_exp2"] = V
    out["dIdV_exp2"] = dIdV_exp2
    return out # return output


def write_single(args,out):
    """Write data for a single iteration"""
    args = dict2obj(args) # convert to object
    # input data
    np.savetxt(args.dIdV_input,np.array([out["V_exp"],out["dIdV_exp"]]).T)
    # tip DOS
    np.savetxt(args.tip_output,np.array([out["V_tip"],out["dos_tip"]]).T)
    # thermal tip DOS
    np.savetxt("THERMAL_"+args.tip_output,np.array([out["V_tipT"],out["dos_tipT"]]).T)
    # DOS
    np.savetxt(args.output,np.array([out["V_dos"],out["dos"]]).T)
    # reconvolved dIdV
    print("dIdV with deconvoluted DOS written to ",args.dIdV_output)
    np.savetxt(args.dIdV_output,np.array([out["V_exp2"],out["dIdV_exp2"]]).T)


def string2dict(l):
    """Transform a string of bash inputs into an object with attributes"""
    out = default_dict() # default dictionary
    ls = l.split() # split
    for i in range(len(ls)//2):
        key = ls[2*i].replace("--","")
        atr = ls[2*i+1]
        out[key] = atr # store
    return out




def default_dict():
    """Generate default dictionary"""
    out = dict()
    out["input"] = "data.cur"
    out["output"] = "DECONVOLUTED_DOS.OUT"
    out["tip_output"] = "TIP_DOS.OUT"
    out["dIdV_output"] = "dIdV_OUTPUT.OUT"
    out["dIdV_input"] = "dIdV_INPUT.OUT"
    out["delta"] = 0.12
    out["Ttip"] = 0.02
    out["Tsur"] = 0.02
    out["Tdos"] = 0.02
    out["maxn"] = 100
    out["ntries"] = 2
    out["gamma"] = 0.01
    out["mode"] = "minimize"
    return out



