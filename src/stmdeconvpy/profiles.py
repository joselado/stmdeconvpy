import numpy as np

def superconducting(delta=0.1):
    """
    Return a BCS-like superconducting filder
    """
    def f(x):
        y = np.zeros(x.shape) # output
        for i in range(len(x)):
            if np.abs(x[i])>delta:
                y[i] = delta/np.sqrt(x[i]**2-delta**2)
            else: 
                y[i] = 0.0 # zero vector
        return y
    return f # superconducting filter




def random_peaks(nmin=1,nmax=None,wmin=0.1,wmax=None,
        xmin=0.0,xmax=None):
    """
    Return a function that creates a profile with random peaks
    """
    if wmax is None: wmax = wmin
    if nmax is None: nmax = nmin
    if xmax is None: xmax = xmin
    n = np.random.randint(nmin,nmax+1) # number of peaks
    dw = wmax-wmin
    dx = xmax-xmin
    wis = [np.random.random()*dw+wmin for i in range(n)]
    xis = [np.random.random()*dx+xmin for i in range(n)]
    def f(x):
        y = np.zeros(x.shape) # output
        for (wi,xi) in zip(wis,xis): # loop over centers
            y += wi/(wi**2+(x-xi)**2)
        return y
    return f





