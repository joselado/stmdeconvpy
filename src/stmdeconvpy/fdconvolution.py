

from numba import jit

@jit
def fdconv(y1,y2,yfd):
    """Perform the convolution with the Fermi Dirac function"""
    out = [0.0 for i in range(len(y1))]
    for i in range(len(y1)):
      for j in range(-i,len(y1)-i):
          out[j] = out[j] + y1[i]*y2[i+j]*(yfd[i]-yfd[i+j])
    return out




