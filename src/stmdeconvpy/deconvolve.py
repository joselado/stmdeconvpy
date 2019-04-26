import numpy as np
from .profiles import normalize,derivative,interpolate
from . import profiles
from scipy.interpolate import interp1d
from scipy.optimize import minimize,anderson


def best_deconvolve(x1,yexp,x2,ytip,ntries=4,**kwargs):
    """Return the best deconvolution"""
    def f():
        return deconvolve(x1,yexp,x2,ytip,return_error=True,**kwargs)
    (x0,y0,e0) = f() # do it once
    for i in range(ntries-1): # loop
        (x,y,e) = f() # perform the minimization
        if e<e0:
            e0 = e 
            x0 = x
            y0 = y
    print("Best deconvolution out of",ntries,"has error",e0)
    return (x0,y0)



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
#      if sgfilter: sol = smoothen(sol)
    print("Error in this minimization",error)
    if return_error: return x,sol,error
    else: return x,sol


def smoothen(sol):
    """Smoothen a signal"""
    from scipy.signal import savgol_filter
    dn = max([2*(len(sol)//50)+1,3])
#    dn = 5
    return savgol_filter(sol,dn,3)




def single_deconvolve(x1,yexp,x2,ytip,n=41,fd=None,sol=None,
        return_error=True,print_error=True):
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
    if fd is not None: fdx = fd(xs) # evaluate the function
    else: fdx = None
    def fdiff(y): # function with the difference
        out = fdconvolution(xs,y,ytipn,fd=fdx) # special convolution
        diff = out - yexpn # difference
#        print(np.max(np.abs(diff)))
        return diff
    def f(y):
        dd = fdiff(y)
        error = np.mean(np.abs(dd)) # compute the error
        if print_error: print(error)
        return error
    bounds = [(0,1) for ix in xs]
    if sol is None: x0 = np.random.random(len(xs))
    #derivative(xs,yexpn) # default try
    else: x0 = interpolate(x1,sol)(xs)
    res = minimize(f,x0,method="SLSQP",bounds=bounds,
            options={"ftol":1e-10,"maxiter":10000})
#    res = minimize(f,res.x,method="Powell")
    yout = interpolate(xs,res.x)(x1)
    if return_error:  return (x1,yout,f(res.x))
    else:  return (x1,yout)



def deconvolve_exact(x1,yexp,x2,ytip,fd=None,sol=None):
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



def fdconvolution(x,y1,y2,fd=None):
    """Fermi Dirac convolution"""
    if fd is None: # no function given
        return np.convolve(y1,y2,mode="same")/len(y1)
        f1 = interpolate(x,y1) # interpolate
        f2 = interpolate(x,y2) # interpolate
        dx = max(x) - min(x)
        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*20) # as many points
        y3 = np.convolve(f1(xs),f2(xs),mode="same")
        return interpolate(xs,y3)(x)
    else: # assume that there is a Fermi Dirac distribution as input 
#        n = len(x)
#        xn,y1n = expand(x,y1)
#        xn,y2n = expand(x,y2)
        from .fdconvolution import fdconv
#        out = fdconv(y1n,y2n,fd(xn))
        if callable(fd): fdx = fd(x) # call function
        else: fdx = fd # assume it is an array
        return fdconv(y1,y2,fdx)/len(x)
#        out = np.convolve(y1n*fd(xn),y2n,mode="same") 
#        out += - np.convolve(y1n,y2n*fd(xn),mode="same")
#        return out[n:2*n]
#        f1 = interpolate(x,y1) # interpolate
#        f2 = interpolate(x,y2) # interpolate
#        dx = max(x) - min(x)
#        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*3) # as many points
#        out = np.convolve(f1(xs)*fd(xs),f2(xs),mode="same") # first term
#        out += -np.convolve(f1(xs),f2(xs)*fd(xs),mode="same") # second term
#        out = interpolate(xs,out,mode="linear")(x)
#        out = only_positive_derivative(x,out) # retain only positive derivative
#        return out 




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



def convolve_dos(x1,y1,x2,y2,fd=None,n=None,T=0.0):
    """
    Convolve two signals
    """
    if fd is None: # no Fermi Dirac distribution given
        fd = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
    f1 = interpolate(x1,y1)
    f2 = interpolate(x2,y2)
    xmax =  np.max(np.abs([np.max(x1),np.max(x2)]))
    xmin = -xmax # same
#    xmin = np.min([np.min(x1),np.min(x2)])
#    xmax = np.max([np.max(x1),np.max(x2)])
    if n is None: n = len(x1) # as many as x1
    xs = np.linspace(xmin,xmax,n,endpoint=True) # as many points
    yc = fdconvolution(xs,f1(xs),f2(xs),fd=fd)
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




def deconvolve_I(x1,y1,x2,y2,fd=None,T=0.0,**kwargs):
    """
    Deconvolve the I VS V signal
    """
    if fd is None: # no Fermi Dirac distribution given
        fd = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
    return best_deconvolve(x1,y1,x2,y2,fd=fd,**kwargs)



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










