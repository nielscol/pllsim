""" Gradient descent optimization using MSE PLL closed loop transfer funtion
    to ideal second order response
"""
import numpy as np
import matplotlib.pyplot as plt
from libpll.optimize import grad_descent, gss
from libpll.tools import timer, fixed_point
from libpll.pncalc import min_ro_pn, tdc_pn, tdc_pn2, bbpd_pn
from libpll.pllcomp import LoopFilterIIRPhase
from copy import copy
import scipy.signal
import json
import scipy.integrate
# open loop jitter transfer functions

KB = 1.38064852E-23

#######################################################################################
#  Transfer functions
#######################################################################################

def _pll_otf(f, _type, k, fz, fp, delay):
    """ Open loop PLL transfer function, uses lumped gain k instead of PLL model parameters
    """
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz + 1)/((s/wp + 1)*(s**_type))
    # return -k*(1j*f/fz + 1)*np.exp(-2j*np.pi*f*delay)/((1j*f/fp+1)*(2*np.pi*f)**_type)
pll_otf = np.vectorize(_pll_otf, otypes=[complex])

def _pll_otf2(f, _type, m, n, kdco, ki, fz, fp, delay):
    """ Open loop PLL transfer function, uses PLL model parameters
    """
    k = kdco*ki*m/float(n)
    return pll_otf(f, _type, k, fz, fp, delay)
    # return -kdco*ki*(m/float(n))*(1j*f/fz+1)*np.exp(-2j*np.pi*f*delay)/((1j*f/fp+1)*(2*np.pi*f)**_type)
pll_otf2 = np.vectorize(_pll_otf2, otypes=[complex])

def _pll_tf(f, _type, k, fz, fp, delay, *args, **kwargs):
    """ Closed loop PLL transfer function, uses lumped gain k instead of PLL model parameters
    """
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz + 1)/(s**_type*(s/wp + 1) + k*np.exp(-s*delay)*(s/wz + 1))
pll_tf = np.vectorize(_pll_tf, otypes=[complex])

def _pll_tf_(f, _type, k, fz, fp, delay):
    """ Closed loop PLL transfer function, uses lumped gain k instead of PLL model parameters
        NOTE: this is the same as pll_tf, but no *args or **kwargs to deal with numpy vectorize issues
    """
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz + 1)/(s**_type*(s/wp + 1) + k*np.exp(-s*delay)*(s/wz + 1))
pll_tf_ = np.vectorize(_pll_tf_, otypes=[complex])

def _pll_tf2(f, _type, m, n, kdco, ki, fz, fp, delay):
    k = kdco*ki*m/float(n)
    return pll_tf(f, _type, k, fz, fp, delay)
pll_tf2 = np.vectorize(_pll_tf2, otypes=[complex])

def _solpf(f, fn, damping):
    """ Second order low pass filter
    """
    wn = 2*np.pi*fn
    w = 2*np.pi*f
    return wn**2/(-w**2 + 2j*damping*wn*w + wn**2)
solpf = np.vectorize(_solpf, otypes=[complex])

def _pi_pll_tf(f, k, fz, delay, *args, **kwargs):
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz+1)/(s**2 + k*np.exp(-s*delay)*(s/wz + 1))
pi_pll_tf = np.vectorize(_pi_pll_tf, otypes=[complex])

def _pi_pll_tf2(f, k, fz, *args, **kwargs):
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*(s/wz+1)/(s**2 + k*(s/wz + 1))
pi_pll_tf2 = np.vectorize(_pi_pll_tf2, otypes=[complex])

def _lf(f, _type, ki, fz, fp):
    """ Loop filter transfer function
    """
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return ki*(s/wz + 1)/((s/wp + 1)*s**(_type-1))
    # return ki*(1j*f/fz+1)/((1j*f/fp+1)*(2j*np.pi*f)**(_type-1))
lf = np.vectorize(_lf, otypes=[complex])

def _calc_k_so_type2(fn, fp, fz):
    """ Calc k for roll-off matched to second order low pass filter
        (for type 2 response)
    """
    wn = 2*np.pi*fn
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    return wz*wn**2/wp
calc_k_so_type2 = np.vectorize(_calc_k_so_type2, otypes=[float])

#######################################################################################
#  BW calculation
#######################################################################################


def bw_solpf(fn, damping):
    def h2(f):
        return abs(solpf(f, fn, damping))**2
    return gss(h2, arg="f", params={}, _min=0, _max=2*fn, target=0.5, conv_tol=1e-10)

def bw_pi_pll(k, fz):
    def h2(f):
        return abs(pi_pll_tf2(f, k, fz))**2
    return gss(h2, arg="f", params={}, _min=0, _max=2*np.sqrt(k), target=0.5, conv_tol=1e-10)

#######################################################################################
# cost functions
#######################################################################################


def cost_solpf(_type, fn, damping, points, delay=0.0, decades=4):
    freqs = np.geomspace(fn*10**(-decades/2), fn*10**(decades/2), int(points))
    prototype = np.log10(np.abs(solpf(freqs, fn, damping)))
    def f(k, fz, fp, plot=False):
        g = pll_tf(f=freqs, _type=_type, k=k, fz=fz, fp=fp, delay=delay)
        if plot:
            plt.clf()
            plt.plot(np.log10(np.abs(solpf(freqs, fn, damping))))
            plt.plot(np.log10(np.abs(g)))
            plt.show()
        return np.mean((prototype - np.log10(np.abs(g)))**2)

    return f

def pll_pn_power_est(tf, tf_params, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax,
                     mode="tdc", sigma_ph=0.1, points=1025, plot=False):
    """ points must be 2**k+1 for some integer k for Romberg integration
    """
    kpn = pn_dco*pn_dco_df**2
    freqs = np.linspace(0, fmax-fmax/points, points)
    fbin = fmax/points
    g = tf(freqs, **tf_params)
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))
    # ntdc = tdc_pn(fclk, n, m, g)
    if mode is"tdc": ntdc = tdc_pn(fclk, n, m, g)
    elif mode is "bbpd": ntdc = bbpd_pn(fclk, n, sigma_ph, g)
    if plot:
        plt.semilogx(freqs[1:], 10*np.log10(ndco[1:]+ntdc[1:]))
        plt.show()
    # print("^^^^^^^^^^^ %f"%(2*scipy.integrate.romb(ndco+ntdc, dx=fbin)))
    # print(sigma_ph, mode)
    return 2*scipy.integrate.romb(ndco+ntdc, dx=fbin)

def pll_pn_power_est2(tf, tf_params, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax, points=1025):
    freqs = np.linspace(0, fmax-fmax/points, points)
    fbin = fmax/points
    g = tf(freqs, **tf_params)
    ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    ndco[np.where(g==1)] = 0
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))
    ntdc = tdc_pn(fclk, n, m, g)

    return np.sum(2*(ndco+ntdc))*fbin

#######################################################################################
# Optimizer - minimize settling time
#######################################################################################

def phase_margin(tf_params, fmax):
    """ In degrees
    """
    def cost(f):
        return abs(pll_otf(f, **tf_params))
    fug = gss(cost, arg="f", params={}, _min=0, _max=fmax, target=1)
    return 180+np.angle(pll_otf(fug, **tf_params), deg=True)


def opt_pll_tf_pi_ph_margin(damping, ph_margin, tsettle_tol, fclk):
    """ Optimize tsettle of PI-controller PLL damping for fixed phase margin and damping
        returns : tsettle of PLL with specified damping and phase margin
    """
    k = np.log(tsettle_tol)**2/(damping**2)
    fz = np.sqrt(k)/(2*damping*2*np.pi)
    tf_params = dict(_type=2, k=k, fz=fz, fp=np.inf, delay=0)
    return phase_margin(tf_params, fclk)

@timer
def opt_pll_tf_pi_controller_fast_settling(ph_margin, max_tsettle, tsettle_tol, fclk, oversamp=20):
    """ Optimized PI-controller PLL for phase noise and settling time.
        Subject to maximum settling time constrained by tsettle, tol.
        points=1025 for Romberg integration (2**k+1)
    """
    def cost(damping):
        return opt_pll_tf_pi_ph_margin(damping, ph_margin, tsettle_tol, fclk)
    opt_damping = gss(cost, arg="damping", params={},
                      _min=0, _max=1.0, target=ph_margin, conv_tol=1e-5)
    def cost(tsettle):
        k = np.log(tsettle_tol)**2/(opt_damping**2*tsettle**2)
        fz = np.sqrt(k)/(2*opt_damping*2*np.pi)
        return bw_pi_pll(k, fz)
    opt_tsettle = gss(cost, arg="tsettle", params={},
                      _min=0, _max=max_tsettle, target=fclk/oversamp)
    opt_bw = cost(opt_tsettle)
    # print(opt_damping, opt_tsettle)
    #if opt_tsettle > max_tsettle:
    #    raise Exception("Error: It is not possible to achieve the specified phase margin and lock time. \
    #                    Specified tsettle=%E, actual=%E. Decrease phase margin and try again."%(max_tsettle, opt_tsettle))

    print("For fast settling: opt pi tsettle = %E, damping = %f, bw = %E"%(opt_tsettle, opt_damping,opt_bw))

    return pll_tf_pi_controller(opt_tsettle, tsettle_tol, opt_damping)

#######################################################################################
# Optimizer - fit closed loop TF to match ideal second order
#######################################################################################

@timer
def opt_pll_tf_so_type2(fn, damping, points=40, delay=0.0):
    """ Calculates loop filter based on optimization to match a second order low pass filter
    """
    print("\n********************************************************************************")
    print("* Optimizing PLL open loop transfer function A(f)")
    print("\tfn\t-> %E"%fn)
    print("\tdamping\t-> %f"%damping)
    _type = 2

    k = calc_k_so_type2(fn, fp=2*damping*fn, fz=0.1*damping*fn)
    tf_params = dict(k=k, fp=2*damping*fn, fz=0.1*damping*fn) # initial guess for parameters

    f = cost_solpf(_type, fn, damping, points, delay)

    klast = 2*k
    f_last = np.inf
    """ Algorithm:
        - H_LF(s) = (K/s)*(s/wz+1)/(s/wp+1)
        - Gradient descent pole/zero to minimize error
        - Tune K so solpf and PLL response have same tail behavior
        - iterate until the cost function stops decreasing.
    """
    while f(**tf_params) < f_last:
        f_last = f(**tf_params)
        _tf_params = copy(tf_params)
        tf_params = grad_descent(f, ("fz", "fp"), tf_params, conv_tol=1e-5, deriv_step=1e-10)
        k = calc_k_so_type2(fn=fn, fp=tf_params["fp"], fz=tf_params["fz"])
        tf_params = dict(k=k, fz=tf_params["fz"], fp=tf_params["fp"]) # initial guess for parameters

    tf_params = _tf_params
    tf_params["k"] = float(tf_params["k"])
    tf_params["delay"] = delay
    tf_params["_type"] = _type
    tf_params["damping"] = damping
    tf_params["fn"] = fn
    tf_params["bw"] = bw_solpf(fn, damping)
    tf_params["pz"] = "pz" # tf contains a tunable pole and zero

    print("\n* Optimized open loop gain coeficient, pole/zeros locations:")
    print("\n\t\t k  (s/wz + 1)")
    print("\tA(f) = \t--- ----------")
    print("\t\ts^2 (s/wp + 1)\n")
    for k in ["k","fz","fp","bw"]:
        print("\t%s\t-> %E"%(k,tf_params[k]))

    return tf_params

#######################################################################################
# Design PI-controller PLL to have specified settling time/damping
#######################################################################################

def pll_tf_pi_controller(tsettle, tol, damping, delay=0.0):
    """ Computes response of PLL with PI-controller.
        Open loop : one zero + two poles at zero
        Assumes damping <= 1.0
        Specify damping and desired settling time.
    """
    print("\n********************************************************************************")
    print("* Computing PI-controller PLL")
    print("\ttsettle\t-> %E +/- %e"%(tsettle, tsettle*tol))
    print("\tdamping\t-> %f"%damping)
    k = np.log(tol)**2/(damping**2*tsettle**2)
    wz = np.sqrt(k)/(2*damping)
    _type = 2
    tf_params = {}
    tf_params["k"] = k
    tf_params["fp"] = np.inf
    tf_params["fz"] = wz/(2*np.pi)
    tf_params["delay"] = delay
    tf_params["_type"] = _type
    tf_params["damping"] = damping
    tf_params["bw"] = bw_pi_pll(k, wz/(2*np.pi))
    tf_params["pz"] = "z" # tf contains a tunable zero

    print("\n* Open loop gain coeficient, pole/zeros locations:")
    print("\n\t\t k ")
    print("\tA(f) = \t--- (s/wz + 1)")
    print("\t\ts^2\n")
    for k in ["k","fz","bw"]:
        print("\t%s\t-> %E"%(k,tf_params[k]))

    return tf_params

#######################################################################################
# Optimize PI-controller PLL for phase noise
#######################################################################################


def opt_pll_tf_pi_controller_damping(tsettle, tol, pn_dco, pn_dco_df,
                                     m, n, kdco, fclk, fmax, points=1025,
                                     mode="tdc", sigma_ph=0.1, delay=0.0):
    """ Optimize damping of PI-controller PLL for phase noise with fixed
        settling time
    """
    def cost(damping):
        k = np.log(tol)**2/(damping**2*tsettle**2)
        fz = np.sqrt(k)/(2*damping*2*np.pi)
        tf_params = dict(k=k, fz=fz, delay=delay)
        return pll_pn_power_est(pi_pll_tf, tf_params, pn_dco, pn_dco_df, m, n,
                                kdco, fclk, fmax, points=points, mode=mode,
                                sigma_ph=sigma_ph)

    return gss(cost, arg="damping", params={}, _min=0.01, _max=1.0, target=0.0, conv_tol=1e-5)

def opt_pll_tf_pi_controller_tsettle(damping, tsettle, tol, pn_dco, pn_dco_df,
                                     m, n, kdco, fclk, fmax, points=1025, mode="tdc",
                                     sigma_ph=0.1, delay=0.0):
    """ Optimize tsettle of PI-controller PLL for phase noise with fixed
        damping
    """
    def cost(tsettle):
        k = np.log(tol)**2/(damping**2*tsettle**2)
        fz = np.sqrt(k)/(2*damping*2*np.pi)
        tf_params = dict(k=k, fz=fz, delay=delay)
        return pll_pn_power_est(pi_pll_tf, tf_params, pn_dco, pn_dco_df, m, n,
                                kdco, fclk, fmax, points=points, mode=mode,
                                sigma_ph=sigma_ph)

    return gss(cost, arg="tsettle", params={}, _min=0.01*tsettle, _max=tsettle, target=0.0, conv_tol=1e-10)

@timer
def opt_pll_tf_pi_controller(tsettle, tol, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax,
                             delay=0.0, points=1025, mode="tdc", sigma_ph=0.1):
    """ Optimized PI-controller PLL for phase noise and settling time.
        Subject to maximum settling time constrained by tsettle, tol.
        points=1025 for Romberg integration (2**k+1)
    """
    tsettle_min = 0.01*tsettle # have to constrain, 0 will cause div-by-0

    def cost(tsettle):
        opt =  opt_pll_tf_pi_controller_damping(tsettle, tol=tol, pn_dco=pn_dco, pn_dco_df=pn_dco_df,
                                                m=m, n=n,kdco=kdco, fclk=fclk, fmax=fmax, points=points,
                                                mode=mode, sigma_ph=sigma_ph, delay=delay)
        k = np.log(tol)**2/(opt**2*tsettle**2)
        fz = np.sqrt(k)/(2*opt*2*np.pi)
        tf_params = dict(k=k, fz=fz, delay=delay)
        if fz > fmax:
            raise Exception("Please increase fmax of loop filter optimization, frequency of TF zero in optimization exceeded fmax.")

        return pll_pn_power_est(pi_pll_tf, tf_params, pn_dco, pn_dco_df, m, n,
                                kdco, fclk, fmax, points=points, mode=mode, sigma_ph=sigma_ph)
    opt_tsettle = gss(cost, arg="tsettle", params={},
                      _min=tsettle_min, _max=tsettle, target=0.0, conv_tol=1e-5)
    opt_damping = opt_pll_tf_pi_controller_damping(opt_tsettle, tol, pn_dco, pn_dco_df, m, n,
                                                   kdco, fclk, fmax, points=points, mode=mode,
                                                   sigma_ph=sigma_ph, delay=delay)

    print("opt pi tsettle = %E, damping = %f"%(opt_tsettle, opt_damping))
    return pll_tf_pi_controller(opt_tsettle, tol, opt_damping, delay=delay)

#######################################################################################
# Optimize BB-PD PI-controller PLL for phase noise
#######################################################################################


@timer
def opt_pll_tf_pi_controller_bbpd(tsettle, tol, pn_dco, pn_dco_df, n, kdco, fclk,
                                  fmax, delay=0, points=1025, max_iter=15):
    """ This does not work yet
    """
    sigma_ph = 0.01
    def cost(sigma_ph):
        m = 2*np.pi
        tf_params = opt_pll_tf_pi_controller(tsettle, tol, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax,
                                             points=points, mode="bbpd", sigma_ph=sigma_ph, delay=delay)
        _sigma_ph2 = pll_pn_power_est(pi_pll_tf, tf_params, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax,
                                    mode="bbpd", sigma_ph=sigma_ph, points=1025)
        print(sigma_ph, np.sqrt(_sigma_ph2)/n, (sigma_ph-np.sqrt(_sigma_ph2)/n)**2)
        return (sigma_ph-np.sqrt(_sigma_ph2)/n)**2
    sigma_ph = gss(cost, arg="sigma_ph", params={}, _min=0.0, _max=1/n, max_iter=max_iter)

    print("opt sigma_ph", sigma_ph)
    m = 2*np.pi
    tf =  opt_pll_tf_pi_controller(tsettle, tol, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax,
                                   points=points, mode="bbpd", sigma_ph=sigma_ph, delay=delay)
    return tf, sigma_ph

def opt_pll_pzk(tsettle, tol, pn_dco, pn_dco_df, m, n, kdco, fclk, fmax, points):
    """ Optimize PLL with one adjustable pole, one adjustable zero and adjustable gain in open loop
        i.e. open loop A(f) = (k/s^2)*(s/wz+1)/(s/wp+1)
        Subject to maximum settling time constrained by tsettle, tol.
    """
    tsettle_min = 0.01*tsettle

    def cost():
        pass

#######################################################################################
# Compute loop filter design from closed loop PLL transfer function parameters
######################################################################################

@timer
def lf_from_pll_tf(tf_params, m, n, kdco, fclk):
    print("\n********************************************************************************")
    print("* Computing loop filter coefficients from OTF")
    print("\n* Input parameters:")
    for k in ["k","fz","fp"]:
        print("\t%s\t-> %E"%(k,tf_params[k]))

    T = 1/float(fclk)
    tf_params["ki"] = tf_params["k"]*n/(m*kdco)
    tf_params["kp"] = tf_params["ki"]/(2*np.pi*tf_params["fz"])
    kp = tf_params["kp"]
    ki = tf_params["ki"]
    wz = 2*np.pi*tf_params["fz"]
    wp = 2*np.pi*tf_params["fp"]

    print("\n* Calculated PLL model parameters:")
    for k in ["ki","kp"]:
        print("\t%s\t-> %E"%(k,tf_params[k]))

    if tf_params["pz"] is "pz":
        tf_params["a0"] = a0 = ki*(wp/wz)*T*(1+wz*T)/(1+wp*T)
        tf_params["a1"] = a1 = -ki*(wp/wz)*T/(1+wp*T)
        tf_params["b0"] = b0 = 1
        tf_params["b1"] = b1 = -(2+wp*T)/(1+wp*T)
        tf_params["b2"] = b2 = 1/(1+wp*T)
    elif tf_params["pz"] is "z":
        # tf_params["a0"] = a0 = kp*(1+wz*T)
        tf_params["a0"] = a0 = kp*(1+wz*T/2)
        # tf_params["a1"] = a1 = -kp
        tf_params["a1"] = a1 = -kp*(1-wz*T/2)
        tf_params["b0"] = b0 = 1
        tf_params["b1"] = b1 = -1
        tf_params["b2"] = b2 = 0
    tf_params["n"] = n
    tf_params["m"] = m
    tf_params["kdco"] = kdco
    tf_params["fclk"] = fclk

    print("\n* Loop filter difference equation")
    print("\ty[n] =\ta0*x[n] + a1*x[n-1] - b1*y[n-1] - b2*y[n-2]")
    # print("\ty[n] =\t%.10E*x[n] + %.10E*x[n-1]\n\t\t+ %.10E*y[n-1] + %.10E*y[n-2]"%(a0,a1,-b1,-b2))
    print("\ta0\t-> %.10E"%a0)
    print("\ta1\t-> %.10E"%a1)
    print("\tb1\t-> %.10E"%b1)
    print("\tb2\t-> %.10E"%b2)

    return tf_params

#######################################################################################
# Optimize number of bits n digital loop filter implementation
######################################################################################


def n_int_bits(lf_params):
    """ Determine number of max bits need to represent integer parts of filter coefficients
    """
    pos_coefs = []
    neg_coefs = []
    for key in ["a0", "a1", "b0", "b1", "b2"]:
        if lf_params[key] is not np.inf:
            if lf_params[key] >= 0.0:
                pos_coefs.append(abs(lf_params[key]))
            else:
                neg_coefs.append(abs(lf_params[key]))
    pos_bits = int(np.floor(np.log2(max(np.abs(np.floor(pos_coefs))))))+1
    neg_bits = int(np.ceil(np.log2(max(np.abs(np.floor(neg_coefs))))))
    return max([pos_bits, neg_bits])

def quant_lf_params(lf_params, int_bits, frac_bits):
    _lf_params = copy(lf_params)
    for key in ["a0", "a1", "b0", "b1", "b2"]:
        if lf_params[key] is not np.inf:
            _lf_params[key] = fixed_point(lf_params[key], int_bits, frac_bits)
    return _lf_params


def var_ntdc_post_lf(lf_params, mode="tdc", steps=513):
    fclk = lf_params["fclk"]
    freqs = np.linspace(0, fclk/2, steps)
    g = pll_tf(freqs, **lf_params)
    ntf_tdc_lf = (lf_params["n"]/lf_params["m"])*(2j*np.pi*freqs/lf_params["kdco"])*g
    if mode is "tdc": npow_det = 1/12
    elif mode is "bbpd": npow_det = (1-2/np.pi)
    return 2*scipy.integrate.romb(abs(ntf_tdc_lf)**2*(1/fclk)*npow_det, 0.5*fclk/steps)

def opt_lf_num_bits(lf_params, min_bits, max_bits, rms_filt_error=0.1, noise_figure=1,
                    sim_steps=1000, fpoints=512, mode="tdc", sigma_ph=0.1):
    """ optimize number of bits for a digital direct form-I implementation using two's complement
        representation fixed point words with all parts of the data path with same data representation
        args:
            noise_figure: the maximum dB increase in noise due to loop filter quantization
            rms_filt_error : RMS value in dB for allowable filter error
    """
    print("\n********************************************************")
    print("Optimizing loop filter digital direct form-I implementation for")
    print("number of bits in fixed point data words utilized")
    sign_bits = 1
    # fint number of integer bits needed
    int_bits = n_int_bits(lf_params)
    print("\n* Integer bits = %d"%int_bits)

    """ Optimization for quantization noise
    """
    print("\n* Optimizing for quantization noise:")
    # find optimal number of bits for quantization noise
    lf_ideal = LoopFilterIIRPhase(ignore_clk=True, **lf_params)
    w = np.floor(np.random.normal(0, 0.1*lf_params["m"], sim_steps))
    pow_ntdc_post_lf = var_ntdc_post_lf(lf_params, mode=mode) # variance of TDC noise at loop filter

    x_ideal = np.zeros(sim_steps)
    for n in range(sim_steps):
        x_ideal[n] = lf_ideal.update(w[n], 0)

    mses = []
    bit_range = range(min_bits-int_bits-1, max_bits-int_bits)
    for frac_bits in bit_range:
        # use a large number of int bits to avoid overflow. Tuning here is with frac bits as
        lf_quant = LoopFilterIIRPhase(ignore_clk=True, int_bits=32, frac_bits=frac_bits, quant_filt=False, **lf_params)
        x_quant = np.zeros(sim_steps)
        for n in range(sim_steps):
            x_quant[n] = lf_quant.update(w[n], 0)
        mse = np.var(x_ideal-x_quant)
        print("\tN bits = %d\tQuant noise power = %E LSB^2"%(frac_bits+int_bits+sign_bits, mse))
        mses.append(mse)
    n = len(mses)-1
    threshold = (10**(noise_figure/10.0) - 1)*pow_ntdc_post_lf
    print("!&!&&!", threshold, pow_ntdc_post_lf)
    while n>=0:
        if mses[n] > threshold:
            n = n+1 if n < len(mses) - 1 else len(mses) - 1
            break
        n -= 1
    opt_frac_bits_qn = bit_range[n]
    print("* Optimum int bits = %d, frac bits = %d, sign bits = 1, quant noise = %.3f LSB^2"%(int_bits, opt_frac_bits_qn, mses[n]))

    """ Optimization for filter accuracy
    """
    print("\n* Optimizing for filter design accuracy:")
    fmin = 1e2
    fclk = lf_params["fclk"]

    a = [lf_params["a0"], lf_params["a1"]]
    b = [lf_params["b0"], lf_params["b1"], lf_params["b2"]]
    f, h_ideal = scipy.signal.freqz(a, b, np.geomspace(fmin, fclk/2, fpoints), fs=fclk)
    s = 2j*np.pi*f
    l = (lf_params["m"]/lf_params["n"])*lf_params["kdco"]*h_ideal/s
    g = l/(1+l)
    bit_range = range(min_bits-int_bits-1, max_bits-int_bits)
    mses = []
    for frac_bits in bit_range:
        _lf_params = quant_lf_params(lf_params, int_bits, frac_bits)
        a = [_lf_params["a0"], _lf_params["a1"]]
        b = [_lf_params["b0"], _lf_params["b1"], _lf_params["b2"]]
        f, h = scipy.signal.freqz(a, b, np.geomspace(fmin, fclk/2, fpoints), fs=fclk)
        s = 2j*np.pi*f
        l = (_lf_params["m"]/_lf_params["n"])*_lf_params["kdco"]*h/s
        g = l/(1+l)
        # w, h = scipy.signal.freqz(a, b, points)
        mses.append(np.var(20*np.log10(np.abs(h[1:]))-20*np.log10(np.abs(h_ideal[1:]))))
        print("\tN bits = %d\tMSE = %E dB^2"%(frac_bits+int_bits+sign_bits, mses[-1]))
    n = len(mses)-1
    while n>=0:
        if mses[n] > rms_filt_error**2:
            n = n+1 if n < len(mses) - 1 else len(mses) - 1
            break
        n -= 1
    opt_frac_bits_filt_acc = bit_range[n]
    print("* Optimum int bits = %d, frac bits = %d, sign_bits=1, quant noise = %E LSB^2"%(int_bits, opt_frac_bits_filt_acc, mses[n]))

    frac_bits = max(opt_frac_bits_qn, opt_frac_bits_filt_acc)
    print("\n* Optimization complete:")
    print("\tInt bits = %d, frac bits = %d, sign bits = 1"%(int_bits, frac_bits))
    print("\tTotal number bits = %d"%(int_bits+frac_bits+sign_bits))
    return int_bits, frac_bits

# @timer
# def calc_loop_filter(k, fn, damping, m, n, kdco, fclk, points=100, delay=0.0):
#     print("\n***********************************************")
#     print("* Computing loop filter coefficients          *")
#     print("***********************************************")
#     T = 1/float(fclk)
#     init = dict(k=k, fz=0.2*damping*fn, fp=2*damping*fn) # initial guess for parameters
# 
#     f = cost_lf(fn, damping, points, delay)
# 
#     tf_params = grad_descent(f, ("k", "fz", "fp"), init, conv_tol=1e-5, deriv_step=1e-10)
# 
#     tf_params["ki"] = tf_params["k"]*n/(m*kdco)
#     tf_params["kp"] = tf_params["ki"]/(2*np.pi*tf_params["fz"])
#     ki = tf_params["ki"]
#     wz = 2*np.pi*tf_params["fz"]
#     wp = 2*np.pi*tf_params["fp"]
# 
#     print("\n* Computed gain coeficients, pole/zeros locations:")
#     for k,v in tf_params.items():
#         print("%s -> %E"%(k,v))
# 
#     tf_params["a0"] = a0 = ki*(wp/wz)*T*(1+wz*T)/(1+wp*T)
#     tf_params["a1"] = a1 = -ki*(wp/wz)*T/(1+wp*T)
#     tf_params["b0"] = b0 = 1
#     tf_params["b1"] = b1 = -(2+wp*T)/(1+wp*T)
#     tf_params["b2"] = b2 = 1/(1+wp*T)
#     tf_params["delay"] = delay
#     tf_params["damping"] = damping
#     tf_params["fn"] = fn
#     tf_params["bw"] = fn
#     tf_params["n"] = n
#     tf_params["m"] = m
#     tf_params["kdco"] = kdco
#     tf_params["fclk"] = fclk
# 
#     print("\n* Loop filter difference equation")
#     print("y[n] = %.10E*x[n] + %.10E*x[n-1] + %.10E*y[n-1] + %.10E*y[n-2]"%(a0,a1,-b1,-b2))
#     print("a0 = %.10E"%a0)
#     print("a1 = %.10E"%a1)
#     print("b1 = %.10E"%b1)
#     print("b2 = %.10E"%b2)
# 
#     return tf_params

