def interpolator2d(x,y,z,mode=None):
    from scipy.interpolate import griddata
    from scipy.interpolate import NearestNDInterpolator
    def f(p):
        return griddata((x,y), z,p[0:2], method='nearest')
    return f

