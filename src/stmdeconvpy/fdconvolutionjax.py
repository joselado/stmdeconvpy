import jax.numpy as jnp
from jax import jit, grad

@jit
def fdconv_jit(y1, y2, yfd1, yfd2):
    n = len(y1) # number of points
    nc = (n-1) // 2 # central site
    out = jnp.zeros_like(y1)

    def compute_convolution(i):
        wp = jnp.arange(-nc, 2*n-nc) - i + nc
        wp = jnp.clip(wp, 0, n-1)
        
        yi = jnp.where(wp < 0, y2[0], jnp.where(wp >= n, y2[-1], y2[wp]))
        di = jnp.where(wp < 0, yfd2[0], jnp.where(wp >= n, yfd2[-1], yfd2[wp]))
        
        j = jnp.arange(-nc, 2*n-nc)
        yj = jnp.where(j >= n, y1[-1], jnp.where(j < 0, y1[0], y1[j]))
        dj = jnp.where(j >= n, yfd1[-1], jnp.where(j < 0, yfd1[0], yfd1[j]))
        
        return jnp.sum(yj * yi * (-dj + di))

    out = jnp.array([compute_convolution(i) for i in range(n)])
    return out

# compute the gradient
grad_fn = grad(lambda y1: jnp.sum(fdconv_jit(y1, y2, yfd1, yfd2)))



def get_fun_grad(ytipn,fd1x,fd2x,yexpn):
    @jit
    def fdiff(y): # function with the difference
        y = y*y # square of the signal
        out = fdconv_jit(y,ytipn,fd1x,fd2x) # special convolution
        diff = out - yexpn # difference
        diff = jnp.mean(diff*diff) # distance
        return diff
    grad_fdiff = grad(fdiff)
    return fdiff,grad_fdiff


