
import numpy as np


def opencur(name):
    """Open a cur file"""
    l = open(name).read().split("[Header end]")[1] # get the numbers
    namenew = name.replace(".cur",".txt") # new name for the data
    open(namenew,"w").write(l) # write the data
    m = np.genfromtxt(namenew).transpose() # get the data
    ys = np.array([iy for (ix,iy) in sorted(zip(m[0],m[1]))])
    xs = np.sort(m[0])
    return (xs,ys)
#    x = np.linspace(np.min(xs),np.max(xs),len(xs))
#    from .profiles import interpolate
#    return x,interpolate(xs,ys)(x)
