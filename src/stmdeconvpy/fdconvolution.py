
import numpy as np

from numba import jit

@jit
def fdconv(y1,y2,yfd1,yfd2):
    """Perform the convolution with the Fermi Dirac function"""
    n = len(y1) # number of points
    nc = (n-1)//2 # central site
    out = [0.0 for i in range(n)]
    for i in range(n):
      for j in range(-nc,2*n-nc):
          wp = j-i+nc # distance to zero frequency
#          if not 0<=wp<n: continue
#          if not 0<=(j+wp)<n: continue
          if wp<0: yi = y2[0] ; di = yfd2[0]
          elif wp>=n: yi = y2[n-1] ; di = yfd2[n-1]
          else: yi = y2[wp] ; di = yfd2[wp]
          if j>=n: yj = y1[n-1] ; dj = yfd1[n-1]
          elif j<0: yj = y1[0] ; dj = yfd1[0]
          else: yj = y1[j] ; dj = yfd1[j]
#          d = yfd[j]
          out[i] = out[i] + yj*yi*(-dj+di)
#          out[i] = out[i] + y1[j]*y2[j+wp]*(yfd[j]-yfd[j+wp])
#    print(out) ; exit()
#    out = np.array(out)/n # normalize 
    return np.array(out)



@jit
def conv(y1,y2):
    """Perform the convolution with the Fermi Dirac function"""
    n = len(y1) # number of points
    nc = (n-1)//2 # central site
    out = [0.0 for i in range(n)]
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





