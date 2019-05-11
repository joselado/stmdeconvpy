import numpy as np
from .fdconvolution import fdconvolution
from .profiles import interpolate


def single_deconvolve_algebra(x1,yexp,x2,ytip,n=41,sol=None,
        return_error=True,print_error=True,
        fd1=None,fd2=None,**kwargs):
    """
    Deconvolve a signal of the form y1(x)*y2(x-x')dx'
    """
    f1 = interpolate(x1,yexp) # interpolate
    f2 = interpolate(x2,ytip) # interpolate
    xmax =  np.max(np.abs([np.max(x1),np.max(x2)]))
    xmin = -xmax # same
    # create the points centered at zero
    xs = np.linspace(xmin,xmax,n,endpoint=True) # as many points
    yexpn = f1(xs) # interpolated data
    ytipn = f2(xs) # interpolated data
    if fd1 is not None: fd1x = fd1(xs) # evaluate the function
    if fd2 is not None: fd2x = fd2(xs) # evaluate the function
    else: fdx = None
    def LO(y): # function returning the linear action on the DOS
        out = fdconvolution(xs,y,ytipn,fd1=fd1x,fd2=fd2x) # special convolution
        return out
    # create the matrix of the linear operator
    MLO = np.zeros((n,n),dtype=np.complex) # initialize
    for i in range(n): # loop
        v = np.zeros(n) ; v[i] = 1.0 # element of the basis
        w = LO(v)
        MLO[:,i] = w # store
    import scipy.linalg as lg
 #   for i in range(n): MLO[i,i] += 1j*0.01
 #   yout = lg.solve(MLO,yexpn).real
    yout = lg.lstsq(MLO,yexpn,cond=1e-5)[0]
    error = 1.0
    yout = interpolate(xs,yout,positive=True)(x1)
    if return_error:  return (x1,yout,error)
    else:  return (x1,yout)





def single_deconvolve_minimize(x1,yexp,x2,ytip,n=41,sol=None,
        return_error=True,print_error=True,
        fd1=None,fd2=None,**kwargs):
    """
    Deconvolve a signal of the form y1(x)*y2(x-x')dx'
    """
    from scipy.optimize import minimize
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
    yout = interpolate(xs,res.x,positive=True)(x1)
    if return_error:  return (x1,yout,f(res.x))
    else:  return (x1,yout)

