import numpy as np
from .profiles import normalize,derivative,interpolate
from . import profiles
from scipy.interpolate import interp1d
from scipy.optimize import minimize,anderson


def best_deconvolve(x1,yexp,x2,ytip,ntries=4,return_error=False,
        **kwargs):
    """Return the best deconvolution"""
    def f(sol):
        return deconvolve(x1,yexp,x2,ytip,sol=sol,return_error=True,**kwargs)
    # random initial guesses
    sols = [np.random.random(x1.shape) for i in range(ntries)]
    from . import parallel
    parallel.cores = min([parallel.maxcpu,ntries]) # number of cores
    # compute everything
    outs = parallel.pcall(f,sols)
    (x0,y0,e0) = outs[0] # do it once
    for i in range(ntries-1): # loop
        (x,y,e) = outs[i] # perform the minimization
        if e<e0:
            e0 = e 
            x0 = x
            y0 = y
    ys = np.array([outs[i][1] for i in range(ntries)]) # get the fits
    es = np.array([outs[i][2] for i in range(ntries)]) # get the errors
    dy = np.sqrt(np.var(ys,axis=0)) # compute variance
    y0 = np.average(ys,axis=0,weights=1./es) # compute weighted average
    print("Best deconvolution out of",ntries,"has error",e0)
    if return_error: return (x0,y0,dy)
    else: return (x0,y0)



def deconvolve(x1,yexp,x2,ytip,ns=None,return_error=False,
        sol=None,sgfilter=True,n=100,**kwargs):
    """Perform a deconvolution of a signal"""
    if ns is None:
#        ns = [21,41,91,131]
        ni = 2*(n//2) +1 # odd number
        ns = [41,ni] # compute twice
    for ni in ns:
      (x,sol,error) = single_deconvolve(x1,yexp,x2,ytip,n=ni,sol=sol,
              return_error=True,**kwargs)
      if sgfilter: sol = smoothen(sol) # smoothen the solution
    print("Error in this minimization",error)
    if return_error: return x,sol,error
    else: return x,sol


def smoothen(sol):
    """Smoothen a signal"""
    from scipy.signal import savgol_filter
    dn = max([2*(len(sol)//50)+1,3])
#    dn = 5
    return savgol_filter(sol,dn,3)




def single_deconvolve(x1,yexp,x2,ytip,n=41,sol=None,
        return_error=True,print_error=True,
        fd1=None,fd2=None,**kwargs):
    """
    Deconvolve a signal of the form y1(x)*y2(x-x')dx'
    """
    f1 = interpolate(x1,yexp)
    f2 = interpolate(x2,ytip)
#    xmin = np.min([np.min(x1),np.min(x2)])
#    xmax = np.max([np.max(x1),np.max(x2)])
    xmax =  np.max(np.abs([np.max(x1),np.max(x2)]))
    xmin = -xmax # same
    # create the points centered at zero
    xs = np.linspace(xmin,xmax,n,endpoint=True) # as many points
    yexpn = f1(xs) # interpolated data
    ytipn = f2(xs) # interpolated data
    if fd1 is not None: fd1x = fd1(xs) # evaluate the function
    if fd2 is not None: fd2x = fd2(xs) # evaluate the function
    else: fdx = None
    def fdiff(y): # function with the difference
        out = fdconvolution(xs,y,ytipn,fd1=fd1x,fd2=fd2x) # special convolution
        diff = out - yexpn # difference
#        print(np.max(np.abs(diff)))
        return diff
    def f(y):
        dd = fdiff(y)
        # compute the error
        error = np.abs(dd) #+ np.mean(np.abs(np.diff(dd))) 
#        dmax = np.max(dd) # maximum error
        error = np.sqrt(np.mean(error**2)) # define error
        error = np.mean(np.abs(dd)) #+ np.mean(np.abs(np.diff(dd))) 
        if print_error: print(error)
        return error
    bounds = [(0,10) for ix in xs]
    if sol is None: x0 = np.random.random(len(xs))
    #derivative(xs,yexpn) # default try
    else: x0 = interpolate(x1,sol)(xs)
    res = minimize(f,x0,method="SLSQP",bounds=bounds,
            options={"ftol":1e-10,"maxiter":10000})
#    res = minimize(f,res.x,method="Powell")
    yout = interpolate(xs,res.x)(x1)
    if return_error:  return (x1,yout,f(res.x))
    else:  return (x1,yout)


# this is a failed attempt to use linalg to solve the problem
#from .deconvolvelstsq import single_deconvolve



def deconvolve_exact(x1,yexp,x2,ytip,fd1=None,fd2=None,
        sol=None):
    """Use the deconvolution in the cleanest way"""
    bounds = [(0,1) for ix in x1]
    def f(y):
        (xx,yy) = dos2I(x1,y,x2,ytip)
        error = np.sum(np.abs(yy - yexp)**2)
        print(error)
        return error
    if sol is None: x0 = yexp # default try
    else: x0 = interpolate(x1,sol)(x1)
    res = minimize(f,x0,method="SLSQP",bounds=bounds,
            options={"ftol":1e-11,"maxiter":100})
    return (x1,res.x)



#deconvolve = deconvolve_exact



def fdconvolution(x,y1,y2,fd1=None,fd2=None):
    """Fermi Dirac convolution"""
    if fd1 is None and fd2 is None: # no function given
        return np.convolve(y1,y2,mode="same")/len(y1)
        f1 = interpolate(x,y1) # interpolate
        f2 = interpolate(x,y2) # interpolate
        dx = max(x) - min(x)
        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*20) # as many points
        y3 = np.convolve(f1(xs),f2(xs),mode="same")
        return interpolate(xs,y3)(x)
    else: # assume that there is a Fermi Dirac distribution as input 
        from .fdconvolution import fdconv
        if callable(fd1): fd1x = fd1(x) # call function
        else: fd1x = fd1 # assume it is an array
        if callable(fd2): fd2x = fd2(x) # call function
        else: fd2x = fd2 # assume it is an array
        return fdconv(y1,y2,fd1x,fd2x)/len(x)




def expand(x,y):
    n = len(x)
    dx = np.max(x) - np.min(x)
    xi = np.linspace(np.min(x),np.max(x),len(x),endpoint=False)
    xn = np.concatenate([xi-dx,x,xi+dx]) # new x axis
    yn = np.concatenate([np.zeros(n)+y[0],y,np.zeros(n)+y[n-1]])
    return xn,yn



def only_positive_derivative(x,y):
    """Return y only taking positive derivative"""
    f = interpolate(x,y)
    from scipy.misc import derivative
    dy = derivative(f,x, dx=1e-3) # compute the derivative
    dy[dy<0.0] = 0.0 # remove negative derivative
    return integrate(x,dy)



def convolve_dos(x1,y1,x2,y2,n=None,Ttip=0.0,Tsur=0.0,T=None):
    """
    Convolve two signals
    """
    if T is not None:
        fd1 = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
        fd2 = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
    else:
        fd1 = profiles.fermi_dirac(T=Ttip) # Fermi Dirac distribution
        fd2 = profiles.fermi_dirac(T=Tsur) # Fermi Dirac distribution
    f1 = interpolate(x1,y1)
    f2 = interpolate(x2,y2)
    xmax =  np.max(np.abs([np.max(x1),np.max(x2)]))
    xmin = -xmax # same
#    xmin = np.min([np.min(x1),np.min(x2)])
#    xmax = np.max([np.max(x1),np.max(x2)])
    if n is None: n = len(x1) # as many as x1
    xs = np.linspace(xmin,xmax,n,endpoint=True) # as many points
    yc = fdconvolution(xs,f1(xs),f2(xs),fd1=fd1(xs),fd2=fd2(xs))
    yo = interpolate(xs,yc)(x1)
    return (x1,yo) # return



def dos2I(x1,y1,x2,y2,**kwargs):
    """
    Convert the two DOS into a I VS V signal
    """
    return convolve_dos(x1,y1,x2,y2,**kwargs)






def I2dIdV(x,y):
    """COmpute dIdV from I"""
    dy = derivative(x,y) # compute the derivative
#    dy = normalize(dy) # normalize profile
    return (x,dy) # return x and derivative



def dos2dIdV(x1,y1,x2,y2,**kwargs):
    """
    Convert the two DOS into a dIdV VS V signal
    """
    (x,y) = dos2I(x1,y1,x2,y2,**kwargs) # get the I VS V
    return I2dIdV(x,y) # return x and derivative




def deconvolve_I(x1,y1,x2,y2,T=None,Ttip=0.0,Tsur=0.0,**kwargs):
    """
    Deconvolve the I VS V signal
    """
    if T is not None: # one temperature
      fd1 = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
      fd2 = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
    else: # two temperatures
      fd1 = profiles.fermi_dirac(T=Ttip) # Fermi Dirac distribution
      fd2 = profiles.fermi_dirac(T=Tsur) # Fermi Dirac distribution
    return best_deconvolve(x1,y1,x2,y2,fd1=fd1,fd2=fd2,**kwargs)



def integrate(x,y):
    """Integrate an array"""
    out = np.array([np.trapz(y[0:n],x[0:n]) for n in range(len(x))])
#    out = normalize(out)
    return out # retunr array


def deconvolve_dIdV(x1,yexp,x2,ytip,**kwargs):
    """
    Deconvolve a dIdV signal
    """
    yexpi = normalize(integrate(x1,yexp)) # integrate and normalize
#    yexpi = integrate(x1,yexp) # integrate and normalize
#    return (x1,yi)
    (xout,yout) = deconvolve_I(x1,yexpi,x2,ytip,**kwargs)
    return (xout,yout)


def dIdV2I(x,y):
    """Integrate dIdV"""
    yi = integrate(x,y) # integrate
    f = interpolate(x,yi) # interpolate
    yi = normalize(f(x) - f(0)) # set to zero
    return yi/np.max(np.abs(yi))/500


def expand_I(x,y):
    """Expand a I signal to apply the algorithm"""
    f = interpolate(x,y) # interpolate
    dx = np.max(x) - np.min(x)
    xn = np.linspace(np.min(x)-dx/2.,np.max(x)+dx/2.,len(x)*2)
    return (xn,f(xn))










