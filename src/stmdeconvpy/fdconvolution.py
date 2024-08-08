
import numpy as np

from numba import jit

def fdconv(y1,y2,yfd1,yfd2):
    """Perform the convolution with the Fermi Dirac function"""
    n = len(y1) # number of points
    nc = (n-1)//2 # central site
    out = np.zeros(n) # initialize
    return fdconv_jit(y1,y2,yfd1,yfd2,out)

@jit(nopython=True)
def fdconv_jit(y1,y2,yfd1,yfd2,out):
    n = len(y1) # number of points
    nc = (n-1)//2 # central site
    for i in range(n):
      for j in range(-nc,2*n-nc):
          wp = j-i+nc # distance to zero frequency
          if wp<0: yi = y2[0] ; di = yfd2[0]
          elif wp>=n: yi = y2[n-1] ; di = yfd2[n-1]
          else: yi = y2[wp] ; di = yfd2[wp]
          if j>=n: yj = y1[n-1] ; dj = yfd1[n-1]
          elif j<0: yj = y1[0] ; dj = yfd1[0]
          else: yj = y1[j] ; dj = yfd1[j]
          out[i] = out[i] + yj*yi*(-dj+di)
    return out



@jit(nopython=True)
def conv(y1,y2):
    """Perform the convolution with the Fermi Dirac function"""
    n = len(y1) # number of points
    nc = (n-1)//2 # central site
    out = np.zeros(n)
    for i in range(n):
      for j in range(-nc,2*n-nc):
          wp = j-i+nc # distance to zero frequency
#          if not 0<=wp<n: continue
#          if not 0<=(j+wp)<n: continue
          if wp<0: yi = y2[0] 
          elif wp>=n: yi = y2[n-1] 
          else: yi = y2[wp] 
          if j>=n: yj = y1[n-1] 
          elif j<0: yj = y1[0] 
          else: yj = y1[j] 
          out[i] = out[i] + yj*yi
    return np.array(out)




def fdconvolution(x,y1,y2,fd1=None,fd2=None):
    """Fermi Dirac convolution"""
    if fd1 is None and fd2 is None: # no function given
        yout = np.convolve(y1,y2,mode="same")/len(y1)
#        f1 = interpolate(x,y1) # interpolate
#        f2 = interpolate(x,y2) # interpolate
#        dx = max(x) - min(x)
#        xs = np.linspace(min(x)-dx,max(x)+dx,len(x)*20) # as many points
#        y3 = np.convolve(f1(xs),f2(xs),mode="same")
#        return interpolate(xs,y3)(x)
    else: # assume that there is a Fermi Dirac distribution as input 
        if callable(fd1): fd1x = fd1(x) # call function
        else: fd1x = fd1 # assume it is an array
        if callable(fd2): fd2x = fd2(x) # call function
        else: fd2x = fd2 # assume it is an array
#        yout = fdconv(y1,y2,fd1x,fd2x)/len(x)
        from .fdconvolutionjax import fdconv_jax
        from .fdconvolutionjax import fdconv_python
        yout = fdconv_jax(np.array(y1),np.array(y2),np.array(fd1x),np.array(fd2x))/len(x)
#        yout2 = fdconv_python(np.array(y1),np.array(y2),np.array(fd1x),np.array(fd2x))/len(x)
#        print(np.sum(np.abs(yout-yout2))) #; exit()
    return yout




def plain_convolution(x,y1,y2):
    dx = (np.max(x) - np.min(x))/len(x)
    return np.convolve(y1,y2,mode="same")*dx

