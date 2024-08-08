import jax.numpy as jnp
from jax import jit, grad


import numpy as np

from numba import jit as njit


@njit(nopython=True)
def fdconv_python(y1,y2,yfd1,yfd2):
    n = len(y1) # number of points
    out = np.zeros(n)
    nc = (n-1)//2 # central site
    def compute(i):
        js = np.arange(-nc,2*n-nc) # js
        wps = js-i+nc # distance to zero frequency
        js = np.clip(js, 0, n-1)
        wps = np.clip(wps, 0, n-1)
        return np.sum(y2[wps]*y1[js]*(-yfd1[js]+yfd2[wps]))
    for i in range(n): out[i] = compute(i)
    return out




from jax import jit, vmap, lax
import jax
#jax.config.update("jax_enable_x64", True)

def fdconv_jax(y1, y2, yfd1, yfd2):
    y1n = jnp.array(y1.copy(),dtype=jnp.float64)
    y2n = jnp.array(y2.copy(),dtype=jnp.float64)
    yfd1n = jnp.array(yfd1.copy(),dtype=jnp.float64)
    yfd2n = jnp.array(yfd2.copy(),dtype=jnp.float64)
    out = fdconv_jax_v2(y1n, y2n, yfd1n, yfd2n)
    return np.asarray(out)



@jit
def fdconv_jax(y1, y2, yfd1, yfd2):
    n = len(y1) # number of points
    def compute(i):
        n = len(y1) # number of points
        nc = (n-1) // 2 # central site
        js = jnp.arange(-nc, 2*n-nc) # js
        wps = js - i + nc # distance to zero frequency
        js = jnp.clip(js, 0, n-1)
        wps = jnp.clip(wps, 0, n-1)
        return jnp.sum(y2[wps] * y1[js] * (-yfd1[js] + yfd2[wps]))
    out = vmap(compute)(jnp.arange(n))
    return out



grad_fn = grad(lambda y1: jnp.sum(fdconv_jit(y1, y2, yfd1, yfd2)))


@jit
def kinetic1(y,w):
    diffs = y[:-1] - y[1:]  # Compute the differences between consecutive elements
    o = jnp.sqrt(jnp.mean(w*diffs ** 2))  # Sum of squared differences
    return o


@jit
def kinetic2(y,w):
    # Compute the central differences and sum them
    diffs = y[:-2] + y[2:] - 2 * y[1:-1]
    o = jnp.sqrt(jnp.mean(w*diffs**2))
    return o 


@jit
def log_distance(out,yexp,delta=1e-3):
    return jnp.log((out + delta)/(yexp + delta))


@jit
def linear_distance(out,yexp):
    return out - yexp



def get_fun_grad(ytipn,fd1x,fd2x,yexpn,delta=1e-5,weight = None):
    """Return a function and its gradient"""
    if weight is None:
        n = len(ytipn) # length of the array
        k_w1 = np.zeros(n-1) # initialize
        k_w2 = np.zeros(n-2) # initialize
        k_w1[n//4:3*n//4] = 1.0 # only half of it
        k_w2[n//4:3*n//4] = 1.0 # only half of it
        weight = 1.0
    from .deconvolve import distance_mode
    if distance_mode=="linear":
        distance = linear_distance # function to compute the distance
    if distance_mode=="log":
        distance = log_distance # function to compute the distance
    from .deconvolve import kinetic_quench 

    @jit
    def fdiff(y): # function with the difference
        y = y*y # square of the signal
        out = fdconv_jax(y,ytipn,fd1x,fd2x) # special convolution
        diff = distance(out,yexpn) # compute distance vector
        diff = jnp.mean(weight*diff*diff) # distance
        # add the kinetic quench
        diff = diff + kinetic_quench*kinetic1(y,k_w1) #+ kinetic2(y))
        diff = diff + kinetic_quench*kinetic2(y,k_w2) #+ kinetic2(y))
      #  diff = jnp.sqrt(diff)
        return diff
    grad_fdiff = grad(fdiff)
    return fdiff,grad_fdiff
    # now compute the hessian
#    def hess_vec_prod(x, v):
#        hessian = jax.jacrev(jax.grad(fdiff))(x)
#        return hessian @ v
#    hess_fdiff = lambda x, v: np.array(hess_vec_prod(x, v))
#    return fdiff,grad_fdiff,hess_fdiff


