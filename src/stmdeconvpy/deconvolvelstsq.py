import numpy as np


def single_deconvolve(x1,yexp,x2,ytip,n=41,sol=None,
        return_error=True,print_error=True,
        fd1=None,fd2=None,**kwargs):
    """
    Deconvolve a signal of the form y1(x)*y2(x-x')dx'
    """
    from .deconvolve import fdconvolution,interpolate
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
    for i in range(n): MLO[i,i] += 1j*0.01
    yout = lg.solve(MLO,yexpn).real
    error = 1.0
    yout = interpolate(xs,yout)(x1)
    print(yout)
    if return_error:  return (x1,yout,error)
    else:  return (x1,yout)

