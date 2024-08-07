import numpy as np


def normalize(y):
    return y/np.max(np.abs(y))/2.
    return y/np.sqrt(np.sum(y**2))




def noise(y,w=0.0):
    """Add noise to a signal"""
    return y + (np.random.random(len(y))-0.5)*0.2*np.max(np.abs(y))



def constant():
    def f(x):
        y = x*0.0 + 1.0
        return  normalize(y)
    return f



def step_tip(delta=0.2,highest=1.0,lowest=0.1):
    def f(x):
        out = np.zeros(len(x)) + highest
        out[np.abs(x)<delta] = lowest
        return out
    return f



def vortex_state(delta=0.2,peak=2.0,background=1.0):
    """Profile for the state in a vortex"""
    def f(x):
        o = peak*(delta/(x**2+delta))
        o = o + background*(np.tanh((np.abs(x)-3*delta)/delta) + 1.0)/2.
        return o
    return f





def dynes_superconductor(delta=0.1,gamma=0.0):
    """Return the profile of a Dynes superconductor"""
    def f(x):
        d = (x+1j*gamma)**2 - delta**2 # denominator
        phi = np.angle(d) # compute the complex phase
        den2 = np.sqrt(np.abs(d))*np.exp(1j*phi/2) # denominator
        den = den2*0. # initialize
        for i in range(len(den)):
            if den2[i].imag>0.0:  den[i] = den2[i] # positive
            else: den[i] = -den2[i]
        num = x + 1j*gamma # compute numerator
        return np.abs((num/den).real) # return profile
    return f




def superconducting(delta=0.1,T=None):
    """
    Return a BCS-like superconducting filter
    """
    def f(x):
        y = np.zeros(x.shape) # output
        for i in range(len(x)):
            if np.abs(x[i])>delta:
                y[i] = abs(x[i])/np.sqrt(x[i]**2-delta**2)
            else: 
                y[i] = 0.0 # zero vector
        y = normalize(y) # normalize the function
        if T is not None and T!=0.0: 
            y = add_temperature(x,y,T)
            f = interpolate(x,y)
            y = (f(x) + f(-x))/2. # symmetrize
        return y
    return f # superconducting filter



def add_temperature(x,y,T=0.0):
    """Add temperature to a signal"""
    fd = dfd_dE(T=T)(x) # get the points
#    fd = derivative(x,fd) # get the derivative
    from . import fdconvolution
#    return fd
    y = fdconvolution.conv(y,fd)/len(y)
    return y



def derivative(x,y):
    """Compute a derivative"""
    f = interpolate(x,y) # interpolate
    from numba import jit
    @jit(nopython=True)
    def der(y,dx):
        n = len(y)
        yo = np.zeros(n) # output
        for i in range(1,n-1):
            yo[i] = (y[i+1] - y[i-1])
        yo[0] = y[1] - y[0]
        yo[n-1] = y[n-1] - y[n-2]
        return yo/dx
    return der(y,1e-3)
#    from scipy.misc import derivative as der
#    return der(f,x, dx=1e-2) # compute the derivative


from scipy.interpolate import interp1d
def interpolate(x,y,mode="linear",positive=False):
    if positive: # enforce the function to be positive
        y2 = y*1.# enforce the function to be positive
        y2[y2<0.] = 0.
    else: y2 = y
    f = interp1d(x,y2,fill_value=(y[0],y[len(y)-1]),
            bounds_error=False,kind=mode)
    if positive: # enforce the function to be positive
        f2 = lambda x: np.abs(f(x))
    else: f2 = f
    return f


def discard_edge(x,y,r=0.2):
    """Discard the edges of the signal"""
    f = interpolate(x,y,mode="cubic",positive=True)
    dx = r*(np.max(x) - np.min(x))/2.
    xi = np.linspace(np.min(x)+dx,np.max(x)-dx,len(x))
    yc = f(xi) # cropped signal
    fc = interpolate(xi,yc,mode="cubic",positive=True)
    return x,fc(x)





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
            return 1.0/(1.0 + np.exp(x/T))
    return f # return the function


def fermi_dirac_derivative(T=0.0):
    """
    Fermi Dirac distribution
    """
    if T==0.0: raise # zero temperature
    else:
        def f(x): 
            o = np.exp(x/T)/T
            return o/(1.0 + np.exp(x/T))**2
    return f # return the function








def dfd_dE(T=0.0):
    """
    Fermi Dirac distribution
    """
    if T==0.0: raise # zero temperature
    else:
        def f(x): 
            w = np.exp(x/T)
#            return 1.0/(1.0 + w)
            return w/T/(1.0 + w)**2
    return f # return the function


def integrate(x,y):
    yi = integrate_array(x,y) # integrate
    f = interpolate(x,yi) # interpolate
    yi = normalize(f(x) - f(0)) # set to zero
    return yi



def integrate_array(x,y):
    """Integrate an array"""
    out = np.array([np.trapz(y[0:n],x[0:n]) for n in range(len(x))])
    return out # retunr array

