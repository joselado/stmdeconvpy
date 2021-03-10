
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


def openfile(name):
    """Open a file"""
    if name.split(".")[-1]=="cur":
        return opencur(name)
    else:
        out = np.genfromtxt(name).T
        return out[0],out[1]




def crop_uncertainty(x,y,d):
    """Return the same data where the points
    with large uncertainty have been renormalized"""
    w = 1
    cut = (w*np.min(d)+np.max(d))/(w+1) # cutoff uncertainty
    cut = np.max(d)/40
    yo = y.copy() # copy result
    yo = y*(1-np.tanh(10*d/y))
    return (x,yo,d)




def readmap(name):
    """Read a map file"""
    ls = open(name).readlines() # read the file
    f = open("dIdV_INPUT_MAP.OUT","w")
    for i in range(len(ls)): # loop
        if i>3: f.write(ls[i]+"  ")
    f.close()
    m = np.genfromtxt("dIdV_INPUT_MAP.OUT").T # return data
#    np.savetxt(name+".txt",np.array([m[1],m[0],m[2]]).T)
    return m


def mapsplit(name):
    """Split the map file into several"""
    m = readmap(name)
    y = np.unique(m[1]) # get values
    ny = len(y) # number of ny
    nx = len(m[0])//ny # number of ny
    out = [] # empty list
    print("Detected a grid",nx,ny)
    if nx*ny!=len(m[0]): raise
    k= 0
    for i in range(ny):
      o = []
      for j in range(nx):
          o.append([m[0][k],m[1][k],m[2][k]]) # store
          k += 1
      o = np.array(o).T # transpose
      out.append(o) # store
    return out # return list



def griddata(x, y, z, xi, yi, **kwargs):
    from scipy.interpolate import griddata
    xd,yd = np.meshgrid(xi,yi)
    zi = griddata(np.array([x, y]).T, z, np.array([xd, yd]).T, method='linear')
    return zi.T

