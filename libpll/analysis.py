""" PLL Analysis methods
    Cole Nielsen 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg
import scipy.integrate
from libpll._signal import make_signal, freq_to_index
from libpll.optimize import gss
from copy import copy

###############################################################################
# Autoregressive phase noise fit
###############################################################################

def autocorrel(x, maxlags=10):
    rxx = np.zeros(maxlags+1)
    for n in range(maxlags+1):
        if n == 0:
            rxx[n] = np.sum(x*x)
        else:
            rxx[n] = np.sum(x[n:]*x[:-n])
    return rxx

def ar_model(data, p, fs, points=1025):
    data_fd = np.fft.fft(data)
    half_len = int(0.5*len(data))
    power_data = np.sum(np.abs(data_fd[1:half_len])**2)/float(len(data))**2
    gamma = autocorrel(data, maxlags=p)
    a = solve_yule_walker(gamma, p)

    w_min = 1/float(len(data))
    w, h = scipy.signal.freqz([1], a, np.linspace(2*np.pi*w_min, np.pi, points))

    fbin = fs/len(data)
    romb_fbin = (0.5*fs-fbin)/(points-1)
    power_ar = scipy.integrate.romb(np.abs(h)**2, romb_fbin)
    b = [np.sqrt(power_data/power_ar)]
    return b, a

def solve_yule_walker(gamma, p):
    gamma_1_p = gamma[1:p+1]
    toeplitz_gamma = scipy.linalg.toeplitz(gamma[0:p])
    a_1_p = -np.dot(np.linalg.inv(toeplitz_gamma), gamma_1_p.T)
    a = np.ones(p+1)
    a[1:] = a_1_p
    return a

def noise_power_ar_model(signal, fmax, p=100, points=1025, tmin=None, tmax=None):
    if tmin or tmax:
        tmin = tmin if tmin else 0
        tmax = tmax if tmax else len(signal.td)/signal.fs
        nmin = int(round(tmin*signal.fs))
        nmax = int(round(tmax*signal.fs))
        td = signal.td[nmin:nmax]
    else: td = signal.td
    b, a = ar_model(td, p, fs=signal.fs)
    f, h = scipy.signal.freqz(b, a, np.linspace(0, signal.fs/2, points), fs=signal.fs)
    return 2*scipy.integrate.romb(np.abs(h)**2, 0.5*signal.fs/points)



###############################################################################
# Phase noise measurement
###############################################################################

def eval_model_pn(signal, f0, df, fref=None):
    """ evaluate phase noise of of signal by fitting phase noise model
        and then calculating phase noise at df for model
        args:
            signal - Signal class object containing oscillator time domain data
            f0 - oscillator fundamental frequency
            df - phase noise frequency offset from carrier to evaluate
            fref - use to reject reference spurs
        returns:
            phase noise of oscillator signal at frequency offset df
    """
    carrier_index = freq_to_index(signal, abs(f0))
    bin_offset = f0 - carrier_index*signal.fbin
    window = scipy.signal.windows.blackmanharris(len(signal.td))
    BH_LOBE_W = 4
    fd = np.fft.fft(window*signal.td)
    carrier = np.abs(fd[carrier_index])
    ssb_slice = copy(fd[carrier_index+1:int(round(len(signal.td)/2))])
    # print(carrier)
    ssb_slice /= carrier
    ssb_slice_f = (np.arange(len(ssb_slice))+1)*signal.fbin - bin_offset
    if fref:
        n = 1
        while True:
            spur_index = int(round((n*fref+bin_offset)/signal.fbin))-1
            if spur_index+BH_LOBE_W > len(ssb_slice_f):
                break
            ssb_slice[spur_index-BH_LOBE_W:spur_index+BH_LOBE_W+1] = np.zeros(2*BH_LOBE_W+1)
            n += 1
    pn_model = fit_rw_pn(ssb_slice_f, ssb_slice)
    return pn_model(df)

def meas_ref_spur(signal, f0, fref, n=1):
    """ evaluate phase noise of of signal by fitting phase noise model
        and then calculating phase noise at df for model
        args:
            signal - Signal class object containing oscillator time domain data
            f0 - oscillator fundamental frequency
            fref - reference freq
            n - what reference spur to measure
        returns:
            phase noise at peak of reference spur, dBc
    """
    carrier_index = freq_to_index(signal, abs(f0))
    bin_offset = f0 - carrier_index*signal.fbin
    window = scipy.signal.windows.blackmanharris(len(signal.td))
    BH_LOBE_W = 4
    fd = np.fft.fft(window*signal.td)
    carrier = np.abs(fd[carrier_index])
    ssb_slice = copy(fd[carrier_index+1:int(round(len(signal.td)/2))])
    # print(carrier)
    ssb_slice /= carrier
    ssb_slice_f = (np.arange(len(ssb_slice))+1)*signal.fbin - bin_offset
    spur_index = int(round((n*fref+bin_offset)/signal.fbin))-1
    if spur_index+BH_LOBE_W > len(ssb_slice_f):
        raise Exception("Spur outside of simulation frequency span")
    return np.abs(ssb_slice[spur_index])

def average_pn(signal, f0, df, bins):
    """ evaluate phase noise of of signal by fitting phase noise model
        and then calculating phase noise at df for model
        args:
            signal - Signal class object containing oscillator time domain data
            f0 - oscillator fundamental frequency
            df - phase noise frequency offset from carrier to evaluate
            fref - use to reject reference spurs
        returns:
            phase noise of oscillator signal at frequency offset df
    """
    carrier_index = freq_to_index(signal, abs(f0))
    bin_offset = f0 - carrier_index*signal.fbin
    window = scipy.signal.windows.blackmanharris(len(signal.td))
    BH_LOBE_W = 4
    fd = np.fft.fft(window*signal.td)
    carrier = np.abs(fd[carrier_index])
    ssb_slice = copy(fd[carrier_index+1:int(round(len(signal.td)/2))])
    # print(carrier)
    ssb_slice /= carrier
    ssb_slice_f = (np.arange(len(ssb_slice))+1)*signal.fbin - bin_offset
    f_index = int(round((df+bin_offset)/signal.fbin))-1
    return np.mean(20*np.log10(np.abs(ssb_slice[f_index-bins:f_index+bins+1])))

def pn_signal(sim_data, div_n):
    """ Extract phase noise/error signal from pllsim data
    """
    td = div_n*(sim_data["div"].td - sim_data["clk"].td)
    return make_signal(td=td, fs=sim_data["clk"].fs)

def pn_signal2(sim_data, div_n):
    """ Extract phase noise/error signal from pllsim data
    """
    td = sim_data["osc"].td - div_n*sim_data["clk"].td
    return make_signal(td=td, fs=sim_data["clk"].fs)

###############################################################################
# Phase noise signal generation
###############################################################################

def osc_td(f0, fs, samples, k, seed=None, rw=[]):
    """ Compute oscillator time domain signal with random walk phase noise
        args:
            f0 - oscillator fundamental frequency
            fs - 1/tstep of simulation
            samples - length of signal to generate
            k - random walk phase gain
            seed - provide 32b value for predictable sequence
    """
    t = np.arange(samples)/float(fs)
    if not any(rw):
        np.random.seed(seed=seed)
        rw = np.cumsum(np.random.choice([-1,1], samples))
    td = np.sin(k*rw + 2*np.pi*f0*t)
    return make_signal(td=td, fs=fs, autocompute_fd=True)

###############################################################################
# Phase noise signal generation
###############################################################################

def fit_rw_pn(f_list, pn_list):
    """ fits 1/f phase noise model to data
        args:
            f_list - frequencies of data points
            pn_list - phase noise at frequencies in f_list
        returns:
            vectorize function modeling phase noise
    """
    k = np.average(np.abs(pn_list)*f_list)
    def _pn_model(df):
        return k/df
    pn_model = np.vectorize(_pn_model, otypes=[float])
    return pn_model

def find_rw_k(f0, fs, samples, pn_db, df, seed=None, rw_seq=[]):
    """ Fits random phase walk gain parameter k to phase noise data
        args:
            f0 - oscillator fundamental frequency
            fs - 1/tstep of simulation
            samples = length of random walk sequence
            pn_db - target phase noise of oscillator
            df - offset of phase noise measurement
            seed - 32b value for predicable random walk sequence
            rw_seq - use if random walk sequence already calculated
    """
    pn = 10**(pn_db/20)
    def cost(k):
        osc = osc_td(f0, fs, samples, k, seed=seed, rw=rw_seq)
        print(pn, eval_model_pn(osc, f0, df))
        return (pn - eval_model_pn(osc, f0, df))**2

    k = gss(cost, arg="k", params={}, _min=0, _max=f0/fs, conv_tol=1e-2)
    return k

###############################################################################
# Noise/power measurement
###############################################################################

def meas_inband_power(signal, fc, bw):
    if not any(signal.fd): signal.fd = np.fft.fft(signal.td)
    f = (np.arange(signal.samples) - int(signal.samples/2))*signal.fbin
    psd = (np.fft.fftshift(signal.fd)/signal.samples)**2
    fslice = np.where(abs(abs(f)-fc) <= bw/2)
    return np.sum(psd[fslice])

def snr(signal, noise, fc, bw):
    s = meas_inband_power(signal, fc, bw)
    n = meas_inband_power(noise, fc, bw)
    return s/n


###############################################################################
# Settling time analysis
###############################################################################

def est_tsettle_eigenvalue(sys, tol):
    """ Estimate settling time based on approximate time constant found
        from state space representation of system.
    """
    eigvals = np.linalg.eigvals(sys._as_ss().A)
    tau = 1/min(abs(np.real(eigvals)))
    return abs(np.log(tol)*tau)

def pll_tf_to_sys(pll_tf):
    """ Take dictionary defining PLL system and make a scipy.system.lti
        object representing the system
    """
    wp = 2*np.pi*pll_tf["fp"]
    wz = 2*np.pi*pll_tf["fz"]
    k = pll_tf["k"]
    if pll_tf["_type"] == 1:
        num = [k/wz, k]
        den = [1/wp, 1+k/wz, k]
    elif pll_tf["_type"] == 2:
        num = [k/wz, k]
        den = [1/wp, 1, k/wz, k]
    return scipy.signal.TransferFunction(num, den)

def est_tsettle_pll_tf(pll_tf, tol):
    """ Estimate PLL settling time based on eigenvalue-based method
    """
    sys = pll_tf_to_sys(pll_tf)
    return est_tsettle_eigenvalue(sys, abs(tol))

def pll_tf_step(pll_tf, tmax):
    """ Generate signal of step response of PLL tranfer function.
    """
    sys = pll_tf_to_sys(pll_tf)
    t = np.arange(0, tmax, 1/pll_tf["fclk"])
    t, x = scipy.signal.step(sys, T=t, N=len(t))
    return make_signal(td=x, fs=pll_tf["fclk"])

def meas_tsettle(sig, tol, limit):
    """ Measure settling time of signal from time domain data
        Note: ringing will cause inaccuracy in the measurement.
    """
    below_tol = np.abs(limit - sig.td) <= limit*tol
    if not any(below_tol):
        return np.nan

    n = len(sig.td)-1
    while n>0 and below_tol[n]:
        if not below_tol[n-1]:
            break
        n -= 1
    return n/sig.fs

def meas_pn_power(clk_sig, osc_sig, div_n, tmin=0):
    ph_error = div_n*clk_sig.td - osc_sig.td
    if tmin: nmin = int(round(tmin*clk_sig.fs))
    else: nmin = 0
    return np.var(ph_error[nmin:])

def meas_pn_power_pllsim(pllsim_data, div_n, tmin=0):
   return meas_pn_power(pllsim_data["clk"], pllsim_data["osc"], div_n, tmin)

def meas_tsettle_pllsim(pllsim_data, tol_hz):
    tol = (tol_hz/(pllsim_data["params"]["kdco"]))/pllsim_data["params"]["lf_ideal"]
    return meas_tsettle(pllsim_data["lf"], tol, pllsim_data["params"]["lf_ideal"])


###############################################################################
# Frequency measurement
###############################################################################

def meas_inst_freq(signal):
    return make_signal(td=signal.fs*np.diff(signal.td)/(2*np.pi), fs=signal.fs)

###############################################################################
# Vectorization
###############################################################################

def vector_meas(meas, data, meas_params):
    """ Broadcast signal measurement onto list of signals
    """
    results = []
    for d in data:
        results.append(meas(d, **meas_params))
    return results

###############################################################################
# Vectorization
###############################################################################

def phase_margin(tf_params, fmax):
    """ In degrees
    """
    def cost(f):
        return abs(pll_otf(f, **tf_params))
    fug = gss(cost, arg="f", params={}, _min=0, _max=fmax, target=1)
    return 180+np.angle(pll_otf(fug, **tf_params), deg=True)
