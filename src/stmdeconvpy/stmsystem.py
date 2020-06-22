import numpy as np
from scipy.interpolate import interp1d
from . import profiles

class STMSystem():
    def __init__(self):
        pass
    def set_tip(self,x,y,T=0.0):
        """Set the density of states of the tip"""
        self.dos_tip = profiles.interpolate(x,y) # DOS of the tip  
        self.T_tip = T # temperature of the Tip

