import os
import sys
import numpy as np
path = os.path.dirname(os.path.realpath(__file__)) + "/../bin/"
#mainpath = os.path.dirname(os.path.realpath(__file__)) + "/../src/"
#sys.path.append(mainpath) # add library

import qtwrap # import the library with simple wrappers
app = qtwrap.App() # this is the main interface


def pyexecute(name):
    """Execute a python script"""
    os.system("python "+path+name)


def show_tip_dos():
    """Show the Tip DOS"""
    delta = str(get_energy("delta"))
    gamma = str(get_energy("gamma"))
    args = " --delta "+ delta +" --gamma "+gamma
    args += " --Vwindow "+ str(get_energy("ewindow"))
    os.system("python "+path+"/stmdeconvpy-tipdos"+args+" &")
    clean_data() # clean the text data
    set_data_tip_dos()


def deconvolve():
    """Perform the deconvolution"""
    args = " " # initialize inputs for the script
    args += " --input " + app.getbox("box_input_file") # input file
    args += " --delta " + str(get_energy("delta")) # input file
    args += " --gamma " + str(get_energy("gamma")) # input file
    args += " --maxn " + str(int(app.get("maxn"))) # input file
    args += " --ntries " + str(int(app.get("ntries")))
    args += " --Ttip " + str(get_energy("Ttip"))
    args += " --Tsur " + str(get_energy("Tsur"))
    args += " --show false "
    clean_data() # clean the text data
    if app.getbox("box_mode")=="Algebra":
      args += " --mode algebra "
    if app.getbox("box_input_dimension")=="1D":
      pyexecute("stmdeconvpy-single"+args+" ")
      set_data_dos()
      set_data_input()
    elif app.getbox("box_input_dimension")=="2D":
      pyexecute("stmdeconvpy2d"+args+" ")
      set_data_dos2d()
      set_data_input2d()
    # save the data
    set_data_tip_dos()
    show_deconvolution()

def show_deconvolution():
    if app.getbox("box_input_dimension")=="1D":
      os.system("stmdeconvpy-show-deconv  &")
    elif app.getbox("box_input_dimension")=="2D":
      pyexecute("stmdeconvpy_plotmap  &")
    else: raise

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
    cs = ["Ttip","Tsur","delta","gamma","ewindow"]
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
    clean_data()
    if app.getbox("box_input_dimension")=="1D":
      os.system("stmdeconvpy-input "+args+" ")
    elif app.getbox("box_input_dimension")=="2D":
      pyexecute("stmdeconvpy-input2d "+args+" ")
#    set_data_input()



def get_inputs():
    """Get all the files that are possible inputs"""
    fs = os.listdir(os.getcwd()) # all the files
    out = []
    forms = [".cur",".txt"]
    for f in fs:
        for fm in forms:
            if fm in f: out.append(f) # store
    cb = getattr(app,"box_input_file")
    cb.clear() # clear the items
    cb.addItems(out)



def set_data(label,name):
    """Add the data of the Tip DOS to the window"""
    app.settext(label,open(name).read())

def clean_data():
    """Clean all the text data"""
    app.settext("text_tip_dos","")
    app.settext("text_input","")
    app.settext("text_dos","")

# define the different functions
set_data_tip_dos = lambda: set_data("text_tip_dos","TIP_DOS.OUT")
set_data_dos = lambda: set_data("text_dos","DECONVOLUTED_DOS.OUT")
set_data_dos2d = lambda: set_data("text_dos","DECONVOLUTED_DOS_MAP.OUT")
set_data_input = lambda: set_data("text_input","dIdV_INPUT.OUT")
set_data_input2d = lambda: set_data("text_input","dIdV_INPUT_MAP.OUT")


set_units(app) # set the units
get_inputs() # get the possible input files


signals = dict() # functions to call
signals["show_tip_dos"] = show_tip_dos
signals["show_input"] = show_input
signals["deconvolve"] = deconvolve
signals["show_deconvolution"] = show_deconvolution
app.connect_clicks(signals)
# now initialize the interface
app.run()

