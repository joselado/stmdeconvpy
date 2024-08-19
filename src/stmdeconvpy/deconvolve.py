import numpy as np
from .profiles import normalize,derivative,interpolate
from . import profiles
from scipy.optimize import minimize,anderson

distance_mode = "log"
distance_mode = "linear"
fit_mode = "signal"
kinetic_quench = 1e-3


def best_deconvolve(x1,yexp,x2,ytip,ntries=1,return_error=False,
        **kwargs):
    """Return the best deconvolution"""
    def f(sol):
        return deconvolve(x1,yexp,x2,ytip,sol=sol,return_error=True,**kwargs)
    # random initial guesses
    sols = [np.random.random(x1.shape) for i in range(ntries)]
    from . import parallel
    parallel.cores = min([parallel.maxcpu,ntries]) # number of cores
    parallel.cores = 1
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



def deconvolve(x1,yexp,x2,ytip,ns=None,return_error=False,crop=True,
        sol=None,sgfilter=False,n=200,mode="algebra",**kwargs):
    """Perform a deconvolution of a signal"""
    if ns is None:
        ni = 2*(n//2) +1 # odd number
        ns = [ni] #[41,ni] # compute twice
    if mode=="minimize":
      from .deconvmode import single_deconvolve_minimize as single_deconvolve
    elif mode=="algebra":
      from .deconvmode import single_deconvolve_algebra as single_deconvolve
    else: raise
    for ni in ns:
      (x,sol,error) = single_deconvolve(x1,yexp,x2,ytip,n=ni,sol=sol,
              return_error=True,**kwargs)
#    if crop: x,sol = profiles.discard_edge(x,sol,r=0.1)
#    if sgfilter: sol = smoothen(sol) # smoothen the solution
    print("Error in this minimization",error)
    if return_error: return x,sol,error
    else: return x,sol


def smoothen(sol):
    """Smoothen a signal"""
    sol = sol**2
    from scipy.signal import savgol_filter
    dn = max([2*(len(sol)//50)+1,3])
#    dn = 5
    out = savgol_filter(sol,dn,3)
    out = np.abs(out)
    return np.sqrt(out)



# this is a failed attempt to use linalg to solve the problem



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



def expand(x,y):
    n = len(x)
    dx = np.max(x) - np.min(x)
    xi = np.linspace(np.min(x),np.max(x),len(x),endpoint=True)
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



def convolve_single_dfd(x,y,T=0.0):
    """Convolve a signal with the derivative of the Fermi Dirac distribution"""
    if T==0.0: return x,y
    else:
      fd = profiles.fermi_dirac_derivative(T=T)(x) # Fermi Dirac distribution
      from .fdconvolution import plain_convolution
      yo = plain_convolution(x,y,fd)
      return x,yo



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
    if n is None: n = len(x1) # as many as x1
    xs = x1
#    xs = np.linspace(xmin,xmax,n,endpoint=True) # as many points
    from .fdconvolution import fdconvolution
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
    dy[dy<0.] = 0.
#    dy = normalize(dy) # normalize profile
    return (x,dy) # return x and derivative



def dos2dIdV(x1,y1,x2,y2,**kwargs):
    """
    Convert the two DOS into a dIdV VS V signal
    """
#    (x,y) = dos2I(x1,y1,x2,y2,**kwargs) # get the I VS V
#    return I2dIdV(x,y) # return x and derivative
#    # use a bigger range
    xmax0,xmin0 = np.max(x1),np.min(x1) # max values
    fac = 4
    xmax = fac*xmax0
    xmin = fac*xmin0
    xnew = np.linspace(xmin,xmax,len(x1)*fac,endpoint=True) # new x axis
    y1new = interpolate(x1,y1)(xnew) # in increased range
    y2new = interpolate(x1,y2)(xnew) # in increased range
#    xnew,y1new,y2new = x1,y1,y2
    (x,y) = dos2I(xnew,y1new,xnew,y2new,**kwargs) # get the I VS V
    (xo,yo) = I2dIdV(x,y) # return x and derivative
    return x1,interpolate(xo,yo)(x1)

def dIdV2dos(V_exp,dIdV_exp,V_tip,dos_tip,**kwargs):
    """Compute several times adding a little bit of noise"""
#    print(np.mean(dIdV_exp),np.mean(dos_tip))
    xmax0,xmin0 = np.max(V_exp),np.min(V_exp) # max values
    fac = 2
    xmax = fac*xmax0
    xmin = fac*xmin0
    xnew = np.linspace(xmin,xmax,len(V_exp)*fac,endpoint=True) # new x axis
    dIdV_exp_new = interpolate(V_exp,dIdV_exp)(xnew) # in increased range
    dos_tip_new = interpolate(V_tip,dos_tip)(xnew) # in increased range
#    return dIdV2dos_single(V_exp,dIdV_exp,V_tip,dos_tip,**kwargs)
    xo,yo = dIdV2dos_single(xnew,dIdV_exp_new,xnew,dos_tip_new,**kwargs)
#    return V_exp,interpolate(xo,yo)(V_exp)
    xo,yo = V_exp,interpolate(xo,yo)(V_exp)
    return xo,yo/np.mean(yo)

def dIdV2dos_single(V_exp,dIdV_exp,V_tip,dos_tip,**kwargs):
    """Compute the DOS from a dIdV"""
    I_exp = dIdV2I(V_exp,dIdV_exp) # integrate the I
    return deconvolve_I(V_exp,I_exp,V_tip,dos_tip,**kwargs)




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










