import numpy as np


def deconvolve(x1,y1,x2,y2,n=100):
    """
    Deconvolve a signal
    """
    from scipy.interpolate import interp1d
    f1 = interp1d(x1,y1,fill_value="extrapolate") # interpolate
    f2 = interp1d(x2,y2,fill_value="extrapolate") # interpolate
    xmin = np.min([np.min(x1),np.min(x2)])
    xmax = np.min([np.max(x1),np.max(x2)])
    xs = np.linspace(xmin,xmax,n) # as many points
    yn1 = f1(xs) # interpolated data
    yn2 = f2(xs) # interpolated data
    yn1 /= np.sqrt(np.sum(yn1**2)) # normalize
    yn2 /= np.sqrt(np.sum(yn2**2)) # normalize
    from scipy.optimize import minimize
    def f(y):
        y = np.abs(y)
        out = np.convolve(y,yn2,mode="same") 
        out /= np.sqrt(np.sum(out**2))
        out -= yn1 # convolution error
        out = np.sum(out*out)
        print(out)
        return out
    bounds = [(0,1) for ix in xs]
    res = minimize(f,yn1,method="SLSQP",bounds=bounds,
            options={"ftol":1e-9})
    yn = res.x
    yn /= np.sqrt(np.sum(yn**2)) # normalize
#    yn = anderson(f,yn2)
    return (xs,yn)


def normalize(y):
    return y/np.sqrt(np.sum(y**2))


