import numpy as np


def normalize(y):
    return y/np.max(np.abs(y))/2.
    return y/np.sqrt(np.sum(y**2))






def constant():
    def f(x):
        y = x*0.0 + 1.0
        return  normalize(y)
    return f



def superconducting(delta=0.1):
    """
    Return a BCS-like superconducting filder
    """
    def f(x):
        y = np.zeros(x.shape) # output
        for i in range(len(x)):
            if np.abs(x[i])>delta:
                y[i] = abs(x[i])/np.sqrt(x[i]**2-delta**2)
            else: 
                y[i] = 0.0 # zero vector
        y = normalize(y) # normalize the function
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
        y = normalize(y)
        return y
    return f



def peak(w=0.1,c=0.0):
    """Return a single peak"""
    return random_peaks(nmin=1,nmax=1,wmin=w,wmax=w,
        xmin=c,xmax=c)



def random_resonances(**kwargs):
    """
    Return random resonances
    """
    f = random_peaks(**kwargs)
    def fout(x): 
        y = f(x)
        y = 1.0 + y/np.max(y)
        y = normalize(y)
        return y
    return fout



def fermi_dirac(T=0.0):
    """
    Fermi Dirac distribution
    """
    if T==0.0: # zero temperature
        def f(x): return (1.0-np.sign(x))/2.
    else:
        def f(x): 
            return 1.0/(1.0 - np.exp(-x/T))
    return f # return the function

