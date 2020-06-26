import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.signal
from copy import copy
from libpll.tools import timer, fixed_point
from libpll.pllcomp import LoopFilterPIPhase
from libpll.plot import razavify

#######################################################################################
# Optimize number of bits n digital loop filter implementation
######################################################################################


def _pll_tf(f, k, fz, delay, *args, **kwargs):
    """ Closed loop PLL transfer function, uses lumped gain k instead of PLL model parameters
    """
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz + 1)/(s**2 + k*np.exp(-s*delay)*(s/wz + 1))
pll_tf = np.vectorize(_pll_tf, otypes=[complex])


def n_int_bits(lf_params):
    """ Determine number of max bits need to represent integer parts of filter coefficients
    """
    pos_coefs = []
    neg_coefs = []
    for key in ["b0", "b1"]:
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
    for key in ["b0", "b1"]:
        if lf_params[key] is not np.inf:
            _lf_params[key] = fixed_point(lf_params[key], int_bits, frac_bits)
    return _lf_params


def var_npd_post_lf(lf_params, mode="bbpd", steps=513):
    fref = lf_params["fref"]
    freqs = np.linspace(0, fref/2, steps)
    g = pll_tf(freqs, **lf_params)
    ntf_pd_lf = (1j*freqs/(lf_params["kpd"]*lf_params["kdco"]))*g
    if mode is "tdc": npow_det = 1/12
    elif mode is "bbpd": npow_det = (1-2/np.pi)
    return 2*scipy.integrate.romb(abs(ntf_pd_lf)**2*(1/fref)*npow_det, 0.5*fref/steps)

def opt_lf_num_bits(lf_params, min_bits, max_bits, rms_filt_error=0.1, noise_figure=1,
                    sim_steps=1000, sim_runs=10, fpoints=512, mode="tdc", sigma_ph=0.1, tdc_in_stdev=1,
                    plot=False):
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
    # generate white noise signal w simulating "regular" activity
    if lf_params["mode"] == "tdc":
        # w = np.floor(np.random.normal(0, 0.1*lf_params["m"], sim_steps))
        w = np.floor(np.random.normal(0, tdc_in_stdev, sim_steps))
    else: # BBPD mode, test sequence is random +/- 1
        # w = np.random.normal(0, np.sqrt((1-2/np.pi)), sim_steps)
        w = np.random.choice((-1,1), sim_steps) # ate

    pow_npd_post_lf = var_npd_post_lf(lf_params, mode=mode) # variance of TDC noise at loop filter

    lf_ideal = LoopFilterPIPhase(ignore_clk=True, **lf_params)
    x_ideal = np.zeros(sim_steps)
    for n in range(sim_steps):
        x_ideal[n] = lf_ideal.update(w[n], 0)

    mses = []
    bit_range = range(min_bits-int_bits-1, max_bits-int_bits)
    for frac_bits in bit_range:
        # use a large number of int bits to avoid overflow. Tuning here is with frac bits as
        runs = np.zeros(sim_runs)
        for m in range(sim_runs):
            lf_quant = LoopFilterPIPhase(ignore_clk=True, int_bits=32, frac_bits=frac_bits, quant_filt=False, **lf_params)
            x_quant = np.zeros(sim_steps)
            for n in range(sim_steps):
                x_quant[n] = lf_quant.update(w[n], 0)
            runs[m] = np.var(x_ideal-x_quant)
            mse = np.average(runs)
        print("\tBits = %d,\t #(sign,int,frac) = (%d,%d,%d), \tQuant noise power = %E LSB^2"%((sign_bits+int_bits+frac_bits), sign_bits,int_bits,frac_bits, mse))
        mses.append(mse)
    threshold = (10**(noise_figure/10.0) - 1)*pow_npd_post_lf
    print("Threshold=%E, PD noise post-LF=%E"%(threshold, pow_npd_post_lf))
    for n,v in enumerate(mses):
        if v < threshold:
            break
    opt_frac_bits_qn = bit_range[n]
    print("* Optimum int bits = %d, frac bits = %d, sign bits = 1, quant noise = %.3f LSB^2"%(int_bits, opt_frac_bits_qn, mses[n]))
    if plot:
        plt.figure(1)
        plt.clf()
        plt.semilogy(np.arange(min_bits, max_bits+1), mses)
        plt.title("RMS Quantization Noise versus Filter Coefficient Bits")
        plt.xlabel("Total bits")
        plt.grid()
        plt.ylabel("DAC LSB$^2$")
        razavify()
        ticks = plt.yticks()[0]
        plt.yticks(ticks, ["10$^{%d}$"%int(round(np.log10(x))) for x in ticks])
        plt.xlim(min_bits, max_bits)
        ticks = plt.xticks()[0]
        plt.xticks(ticks, ["%d"%x for x in ticks])

    #////////////////////////////////////////////////////////////////////////////////////
    """ Optimization for filter accuracy
    """
    print("\n* Optimizing for filter design accuracy:")
    fmin = 1e2
    fref = lf_params["fref"]

    b = [lf_params["b0"], lf_params["b1"]]
    a = [1, -1,]
    f, h_ideal = scipy.signal.freqz(b, a, np.geomspace(fmin, fref/2, fpoints), fs=fref)
    s = 2j*np.pi*f
    l = 2*np.pi*lf_params["kpd"]*lf_params["kdco"]*h_ideal/s
    g = l/(1+l)
    bit_range = range(min_bits-int_bits-1, max_bits-int_bits)
    mses = []
    # print(lf_params["b0"], lf_params["b1"])
    for frac_bits in bit_range:
        _lf_params = quant_lf_params(lf_params, int_bits, frac_bits)
        b = [_lf_params["b0"], _lf_params["b1"]]
        # print(_lf_params["b0"], _lf_params["b1"])
        f, h = scipy.signal.freqz(b, a, np.geomspace(fmin, fref/2, fpoints), fs=fref)
        s = 2j*np.pi*f
        l = 2*np.pi*lf_params["kpd"]*lf_params["kdco"]*h/s
        _g = l/(1+l)
        # mses.append(np.var(20*np.log10(np.abs(h[1:]))-20*np.log10(np.abs(h_ideal[1:]))))
        mses.append(np.var(20*np.log10(np.abs(g[1:]))-20*np.log10(np.abs(_g[1:]))))
        # print("\tN bits = %d\tMSE = %E dB^2"%(frac_bits+int_bits+sign_bits, mses[-1]))
        print("\tBits = %d,\t #(sign,int,frac) = (%d,%d,%d), \tMSE = %E LSB^2"%((sign_bits+int_bits+frac_bits), sign_bits,int_bits,frac_bits, mses[-1]))
    n = len(mses)-1
    for n,v in enumerate(mses):
        if v < rms_filt_error**2:
            break
    opt_frac_bits_filt_acc = bit_range[n]
    print("* Optimum int bits = %d, frac bits = %d, sign_bits=1, quant noise = %E LSB^2"%(int_bits, opt_frac_bits_filt_acc, mses[n]))
    if plot:
        plt.figure(2)
        plt.clf()
        plt.semilogy(np.arange(min_bits, max_bits+1), mses)
        plt.title("MSE Filter Error (dB) versus Filter Coefficient Bits")
        plt.xlabel("Total bits")
        plt.ylabel("MSE [dB$^2$]")
        plt.grid()
        razavify()
        ticks = plt.yticks()[0]
        plt.yticks(ticks, ["10$^{%d}$"%int(round(np.log10(x))) for x in ticks])
        plt.xlim(min_bits, max_bits)
        ticks = plt.xticks()[0]
        plt.xticks(ticks, ["%d"%x for x in ticks])

    frac_bits = max(opt_frac_bits_qn, opt_frac_bits_filt_acc)
    print("\n* Optimization complete:")
    print("\tInt bits = %d, frac bits = %d, sign bits = 1"%(int_bits, frac_bits))
    print("\tTotal number bits = %d"%(int_bits+frac_bits+sign_bits))
    return int_bits, frac_bits
