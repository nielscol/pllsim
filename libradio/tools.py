""" Random helping functions / decorators """
import functools
import time
import numpy as np
from math import pi, sqrt, log, sin
from scipy.special import erfc

SQRT2 = sqrt(2.0)
SQRTLN2 = sqrt(log(2.0))

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print("Finished %r in %.2f seconds"%(func.__name__,run_time))
        return value
    return wrapper_timer

##############################################################################
# Useful mathmatical methods
#############################################################################

def sinx_x(x):
    return 1.0 if x==0 else sin(pi*x)/(pi*x)
v_sinx_x = np.vectorize(sinx_x, otypes=[float])


@timer
def sinx_x_interp(x, factor, span, remove_extra=True):
    # make sinx_x pulse shape
    fir_len = int(factor*span)
    if fir_len % 2 == 0:
        fir_len += 1
    t = (np.arange(fir_len)/float(factor)-span/2)
    fir = v_sinx_x(t)
    fir *= factor/float(sum(fir))
    # upsample signal and apply pulse shape through convolution
    upsampled = np.zeros(len(x)*factor)
    upsampled[np.arange(len(x))*factor] = x
    interpolated = np.convolve(upsampled, fir, mode="full")
    if remove_extra:
        interpolated = interpolated[int((fir_len-1)/2):]
        interpolated = interpolated[:len(x)*factor]
    return interpolated


def q(x):
    return 0.5*erfc(x/SQRT2)


def raised_cos(t, tbit, rolloff):
    tbit=float(tbit)
    if rolloff != 0.0 and abs(tbit/(2.0*rolloff)) == t:
        return (pi/(4.0*tbit))*sinx_x(1/(2.0*rolloff))
    elif (2*rolloff*t/tbit)**2 == 1.0:
        if raised_cos(t*(1+1e-6), tbit,rolloff) < 0.25:
            return 0.0
        else:
            return 0.5
    else:
        return (1.0/tbit)*sinx_x(t/tbit)*np.cos(pi*rolloff*t/tbit)/(1.0-(2*rolloff*t/tbit)**2)

v_raised_cos = np.vectorize(raised_cos, otypes=[float])

