import os
import sys
import numpy as np
path = os.path.dirname(os.path.realpath(__file__)) + "/../bin/"
#mainpath = os.path.dirname(os.path.realpath(__file__)) + "/../src/"
#sys.path.append(mainpath) # add library

import qtwrap # import the library with simple wrappers
app = qtwrap.App() # this is the main interface


def show_tip_dos():
    """Show the Tip DOS"""
    delta = str(get_energy("delta"))
    gamma = str(get_energy("gamma"))
    args = " --delta "+ delta +" --gamma "+gamma
    os.system("python "+path+"/stmdeconvpy-tipdos"+args)


def deconvolve():
    """Perform the deconvolution"""
    args = " " # initialize inputs for the script
    args += " --input " + app.getbox("box_input_file") # input file
    args += " --delta " + str(get_energy("delta")) # input file
    args += " --gamma " + str(get_energy("gamma")) # input file
    args += " --maxn " + str(int(app.get("maxn"))) # input file
    args += " --Ttip " + str(get_energy("Ttip"))
    args += " --Tsur " + str(get_energy("Tsur"))
    if app.getbox("box_mode")=="Algebra":
      args += " --mode algebra "
    os.system("stmdeconvpy-single"+args+" &")



def set_units(form):
    """Add the different units"""
    def set_unit_single(name):
      try: cb = getattr(form,name)
      except:
          print("Combobox",name,"not found")
          return
      cs = ["meV","K","eV"]
      cb.clear() # clear the items
      cb.addItems(cs)
    cs = ["Ttip","Tsur","delta","gamma"]
    for c in cs:  set_unit_single("box_"+c)

def get_energy(name):
    """Get an energy in the right units"""
    c = app.get(name) # get the energy
    unit = app.getbox("box_"+name) # get the units
    if unit=="meV": return c
    elif unit=="K": return c*8.6217*1e-2 # to meV
    elif unit=="eV": return c*1e3 # to meV
    else: raise # raise error


def show_input():
    args = ""
    args += "--input "+app.getbox("box_input_file")
    os.system("stmdeconvpy-input "+args+" &")



def get_inputs():
    """Get all the files that are possible inputs"""
    fs = os.listdir(os.getcwd()) # all the files
    out = []
    for f in fs:
        if ".cur" in f: out.append(f) # store
    cb = getattr(app,"box_input_file")
    cb.clear() # clear the items
    cb.addItems(out)

set_units(app) # set the units
get_inputs() # get the possible input files


signals = dict() # functions to call
signals["show_tip_dos"] = show_tip_dos
signals["show_input"] = show_input
signals["deconvolve"] = deconvolve
app.connect_clicks(signals)
# now initialize the interface
app.run()

