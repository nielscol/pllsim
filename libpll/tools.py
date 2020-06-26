""" Random helping functions / decorators """
import functools
import time
import numpy as np

def fixed_point(x, int_bits, frac_bits):
    """ Forced floating point value to be rounded to a float that
        is equivalent in value to the nearest fixed point representation
        of a given resolution. There is assumed to be one extra bit for sign.
        args:
            x - input to be converter
            int_bits - number of integer bits in fixed point
            frac_bits - number of fractional bits in fixed point
        returns:
            converted float equivalent to fixed point representation
    """
    if int_bits == None and frac_bits == None: return x
    _frac = x - np.floor(x) #if x >= 0 else x - np.ceil(x)
    _int = np.floor(x) #if x>= 0 else np.ceil(x)
    _frac = np.round(_frac*2**frac_bits)/2**frac_bits
    _x = _int + _frac
    # if _x > 2**(int_bits-1)-1: _x = 2**(int_bits-1)-1
    # elif _x < -2**(int_bits-1): _x = -2**(int_bits-1)
    if _x > 2**(int_bits)-2**(-frac_bits): _x = 2**(int_bits)-2**(-frac_bits)
    elif _x < -2**(int_bits): _x = -2**(int_bits)
    return _x


def debias_pn(sig, tlock):
    start = int(np.ceil(sig.fs*tlock))
    sig.td -= np.mean(sig.td[start:])
    return sig

def is_edge_phase(x, x_last):
    """ Detects signal edges (2*pi*n, n=0,1,2,..) in phase domain.
    """
    _x = x/(2*np.pi)
    _x = round(_x - round(_x), 5)
    _x_last = x_last/(2*np.pi)
    _x_last = round(_x_last - round(_x_last), 5)
    if _x == 0.0 or (_x_last < 0.0 and _x > 0.0):
        return True
    else:
        return False


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

def binary_lf_coefs(lf_params, int_bits, frac_bits):
    print("\n* Conversion of filter coefficients to digital words:")
    for key in ["a0", "a1", "b0", "b1", "b2"]:
        if key in lf_params:
            x = lf_params[key]
            sgn = int(x<0)
            if sgn: # convert neg -> pos for conversion step
                x *= -1
            x = int(round(x*2**frac_bits)) # shift fractional to above decimal, round
            x &= int(2**(int_bits+frac_bits)-1) # trunctate to int_bits + frac_bits
            if sgn: # convert pos -> neg
                print('{0:b}'.format(x))
                x = ~x
                print('{0:b}'.format(x))
                x &= int(2**(int_bits+frac_bits+1)-1)
                print('{0:b}'.format(x))
                x += 1
                print('{0:b}'.format(x))
                x &= int(2**(int_bits+frac_bits+1)-1)
            x = [int(x) for x in '{0:b}'.format(x)][::-1]
            # if sgn:
                # x.extend([1]*(int_bits+frac_bits-len(x)))
            # else:
            if not sgn:
                x.extend([0]*(1+int_bits+frac_bits-len(x)))
            # x.append(sgn)
            x = [str(y) for y in x]
            # print(x)
            x = [y for y in x]
            print("%s = %E\t->\t%s = 0b%s"%(key,lf_params[key], key,"".join(x[::-1])))

# d = dict(a0=1.433, a1=2.11, b0=-1.0, b1=1.434, b2=-1.5424)
# binary_lf_coefs(d, 4, 4)


