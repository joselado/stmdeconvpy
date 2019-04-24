import numpy as np

def normalize(y):
    return y/np.max(np.abs(y))/2.
    return y/np.sqrt(np.sum(y**2))

