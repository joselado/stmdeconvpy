import numpy as np
from .profiles import normalize
from . import profiles
from scipy.interpolate import interp1d
from scipy.optimize import minimize,anderson


def deconvolve(x1,yexp,x2,ytip,n=200,fd=None,sol=None):
    """
    Deconvolve a signal of the form y1(x)*y2(x-x')dx'
    """
    f1 = interpolate(x1,yexp)
    f2 = interpolate(x2,ytip)
    xmin = np.min([np.min(x1),np.min(x2)])
    xmax = np.max([np.max(x1),np.max(x2)])
    xs = np.linspace(xmin,xmax,n) # as many points
    yexpn = f1(xs) # interpolated data
    ytipn = f2(xs) # interpolated data
#    yexpn = normalize(yexpn) # normalize the input signal
#    ytipn = normalize(ytipn) # normalize the tip DOS
#    xs = x1
#    yexpn = yexp
#    ytip = ytip
    def fdiff(y): # function with the difference
        out = fdconvolution(xs,y,ytipn,fd=fd) # special convolution
        diff = out - yexpn # difference
#        print(np.max(np.abs(diff)))
        return diff
    def f(y):
        dd = fdiff(y)
        error = np.sum(dd**2) # compute the error
        print(error)
        return error
    bounds = [(0,1) for ix in xs]
    if sol is None: x0 = np.random.random(len(xs))
    #derivative(xs,yexpn) # default try
    else: x0 = normalize(interpolate(x1,sol)(xs))
#    x0 = anderson(fdiff,x0,f_tol=0.01)
#    return (xs,x0)
#    from scipy.optimize import differential_evolution
#    res = differential_evolution(f,bounds)
#    res = minimize(f,x0)
    res = minimize(f,x0,method="SLSQP",bounds=bounds,
            options={"ftol":1e-9,"maxiter":100})
#    res = minimize(f,res.x,method="Powell")
    return (x1,interpolate(xs,res.x)(x1))



def deconvolve_exact(x1,yexp,x2,ytip,fd=None,sol=None):
    """Use the deconvolution in the cleanest way"""
    bounds = [(0,1) for ix in x1]
    def f(y):
        (xx,yy) = dos2I(x1,y,x2,ytip)
        error = np.sum(np.abs(yy - yexp)**2)
        print(error)
        return error
    if sol is None: x0 = yexp # default try
    else: x0 = normalize(interpolate(x1,sol)(x1))
    res = minimize(f,x0,method="SLSQP",bounds=bounds,
            options={"ftol":1e-9,"maxiter":1000})
    return (x1,res.x)



deconvolve = deconvolve_exact



def fdconvolution(x,y1,y2,fd=None):
    """Fermi Dirac convolution"""
    if fd is None: # no function given
        return np.convolve(y1,y2,mode="same")
        f1 = interpolate(x,y1) # interpolate
        f2 = interpolate(x,y2) # interpolate
        dx = max(x) - min(x)
        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*20) # as many points
        y3 = np.convolve(f1(xs),f2(xs),mode="same")
        return interpolate(xs,y3)(x)
    else: # assume that there is a Fermi Dirac distribution as input 
        n = len(x)
        xn,y1n = expand(x,y1)
        xn,y2n = expand(x,y2)
        out = np.convolve(y1n*fd(xn),y2n,mode="same") 
        out += - np.convolve(y1n,y2n*fd(xn),mode="same")
        return out[n:2*n]
        f1 = interpolate(x,y1) # interpolate
        f2 = interpolate(x,y2) # interpolate
        dx = max(x) - min(x)
        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*3) # as many points
        out = np.convolve(f1(xs)*fd(xs),f2(xs),mode="same") # first term
        out += -np.convolve(f1(xs),f2(xs)*fd(xs),mode="same") # second term
        out = interpolate(xs,out,mode="linear")(x)
        out = only_positive_derivative(x,out) # retain only positive derivative
        return out 




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
    xmin = np.min([np.min(x1),np.min(x2)])
    xmax = np.max([np.max(x1),np.max(x2)])
    if n is None: n = len(x1) # as many as x1
    xs = np.linspace(xmin,xmax,n) # as many points
    yc = fdconvolution(xs,f1(xs),f2(xs),fd=fd)
    yo = interpolate(xs,yc)(x1)
    return (x1,yo) # return



def dos2I(x1,y1,x2,y2,**kwargs):
    """
    Convert the two DOS into a I VS V signal
    """
    return convolve_dos(x1,y1,x2,y2,**kwargs)



def interpolate(x,y,mode="cubic"):
    f = interp1d(x,y,fill_value=(y[0],y[len(y)-1]),
            bounds_error=False,kind=mode)
    return f



def dos2dIdV(x1,y1,x2,y2,**kwargs):
    """
    Convert the two DOS into a dIdV VS V signal
    """
    (x,y) = dos2I(x1,y1,x2,y2,**kwargs) # get the I VS V
    f = interpolate(x,y) # interpolate
    from scipy.misc import derivative
    dy = derivative(f,x, dx=1e-3) # compute the derivative
    dy = normalize(dy) # normalize profile
    return (x,dy) # return x and derivative

def derivative(x,y):
    """Compute a derivative"""
    f = interpolate(x,y) # interpolate
    from scipy.misc import derivative as der
    return der(f,x, dx=1e-3) # compute the derivative



def deconvolve_I(x1,y1,x2,y2,fd=None,T=0.0,**kwargs):
    """
    Deconvolve the I VS V signal
    """
    if fd is None: # no Fermi Dirac distribution given
        fd = profiles.fermi_dirac(T=T) # Fermi Dirac distribution
    return deconvolve(x1,y1,x2,y2,fd=fd,**kwargs)



def integrate(x,y):
    """Integrate an array"""
    out = np.array([np.trapz(y[0:n],x[0:n]) for n in range(len(x))])
    out = normalize(out)
    return out # retunr array


def deconvolve_dIdV(x1,yexp,x2,ytip,**kwargs):
    """
    Deconvolve a dIdV signal
    """
    yexpi = normalize(integrate(x1,yexp)) # integrate and normalize
#    return (x1,yi)
    (xout,yout) = deconvolve_I(x1,yexpi,x2,ytip,**kwargs)
    return (xout,yout)





