import numpy as np
import matplotlib.pyplot as plt
from libpll.plot import razavify, plot_pn_ar_model
from libpll.filter import opt_pll_tf_pi_controller, lf_from_pll_tf, pll_tf, pi_pll_tf
from libpll.pllcomp import LoopFilterIIRPhase
from libpll.tools import fixed_point
from libpll._signal import make_signal
import scipy.signal
import scipy.integrate
from copy import copy

MAX_TSETTLE = 50e-6
SETTLE_DF = 10e3
INIT_F_ERR = 12e6
DCO_PN = 10**(-84/10)
DCO_PN_DF = 1e6
TDC_STEPS = 150
DIV_N = 150
KDCO = 10e3
FCLK = 16e6
OPT_FMAX = 16e6
USE_BBPD = True
BBPD_TSU = BBPD_TH = 10e-12

SIM_STEPS = 10000
SIGMA_PN = 0.5

MIN_BITS = 8
MAX_BITS = 16
SIGN_BIT = 1

INT_BITS = 32
FRAC_BITS = 4
MAX_PN_DELTA_LF = 1.0 # dB
var_ratio = 10**(MAX_PN_DELTA_LF/10.0) - 1

opt_tf = opt_pll_tf_pi_controller(MAX_TSETTLE, abs(SETTLE_DF/INIT_F_ERR), DCO_PN, DCO_PN_DF, TDC_STEPS,
                                  DIV_N, KDCO, FCLK, OPT_FMAX)
lf_params = lf_from_pll_tf(opt_tf, TDC_STEPS, DIV_N, KDCO, FCLK)

lf_ideal = LoopFilterIIRPhase(ignore_clk=True, **lf_params)

x = np.floor((TDC_STEPS/(2*np.pi))*np.random.normal(0, SIGMA_PN, SIM_STEPS))
# plt.hist(x, bins=100)
# plt.show()

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


def var_ntdc_post_lf(lf_params, steps=513):
    fclk = lf_params["fclk"]
    freqs = np.linspace(0, fclk/2, steps)
    g = pll_tf(freqs, **lf_params)
    ntf_tdc_lf = (lf_params["n"]/lf_params["m"])*(2j*np.pi*freqs/lf_params["kdco"])*g
    return 2*scipy.integrate.romb(abs(ntf_tdc_lf)**2*(1/(12*fclk)), 0.5*fclk/steps)

def pow_ntdc_post_lf(freqs, lf_params, fclk, steps=513):
    g = pll_tf(freqs, **lf_params)
    ntf_tdc_lf = (lf_params["n"]/lf_params["m"])*(2j*np.pi*freqs/lf_params["kdco"])*g
    return abs(ntf_tdc_lf)**2*(1/(12))


def opt_lf_num_bits(lf_params, min_bits, max_bits, rms_filt_error=0.1, noise_factor=1, sim_steps=1000, fpoints=512):
    """ optimize number of bits for a digital direct form-I implementation using two's complement
        representation fixed point words with all parts of the data path with same data representation
        args:
            noise_factor: the maximum dB increase in noise due to loop filter quantization
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
    # w = np.random.choice([-0.05, 0.05], sim_steps)
    pow_ntdc_post_lf = var_ntdc_post_lf(lf_params) # variance of TDC noise at loop filter

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
        print("\tN bits = %d\tQuant noise power = %.3f LSB^2"%(frac_bits+int_bits+sign_bits, mse))
        mses.append(mse)
    n = len(mses)-1
    threshold = (10**(noise_factor/10.0) - 1)*pow_ntdc_post_lf
    print(threshold)
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
        print("\tN bits = %d\tMSE = %.3f dB^2"%(frac_bits+int_bits+sign_bits, mses[-1]))
    n = len(mses)-1
    while n>=0:
        if mses[n] > rms_filt_error:
            n = n+1 if n < len(mses) - 1 else len(mses) - 1
            break
        n -= 1
    opt_frac_bits_filt_acc = bit_range[n]
    print("* Optimum int bits = %d, frac bits = %d, sign_bits=1, quant noise = %.3f LSB^2"%(int_bits, opt_frac_bits_filt_acc, mses[n]))

    frac_bits = max(opt_frac_bits_qn, opt_frac_bits_filt_acc)
    print("\n* Optimization complete:")
    print("\tInt bits = %d, frac bits = %d, sign bits = 1"%(int_bits, frac_bits))
    print("\tTotal number bits = %d"%(int_bits+frac_bits+sign_bits))
    return int_bits, frac_bits


int_bits, frac_bits = opt_lf_num_bits(lf_params, MIN_BITS, MAX_BITS, rms_filt_error=0.1, noise_factor=1, sim_steps=10000, fpoints=512)

print(int_bits, frac_bits)
foo()




""" Optimization for filter quantization noise
"""
ntdc_post_lf = var_ntdc_post_lf(lf_params)
print("****", ntdc_post_lf, var_ratio)
OPT_INT_BITS = n_int_bits(lf_params)
print(OPT_INT_BITS)
filt_ideal = np.zeros(SIM_STEPS)
for n in range(SIM_STEPS):
    filt_ideal[n] = lf_ideal.update(x[n], 0)

freqs = np.geomspace(1e3, 8e6, 1000)
ntdc = pow_ntdc_post_lf(freqs, lf_params, FCLK)
plt.semilogx(freqs, 10*np.log10(ntdc), label="TDC noise")
mses = []
bit_range = range(MIN_BITS-OPT_INT_BITS-1, MAX_BITS-OPT_INT_BITS)
for frac_bits in bit_range:
    lf_quant = LoopFilterIIRPhase(ignore_clk=True, int_bits=INT_BITS, frac_bits=frac_bits, quant_filt=False, **lf_params)
    filt_quant = np.zeros(SIM_STEPS)
    for n in range(SIM_STEPS):
        filt_quant[n] = lf_quant.update(x[n], 0)

    sig = make_signal(td=filt_quant, fs=FCLK)
    plot_pn_ar_model(sig, fmin=1e3)
    mse = np.var(filt_ideal-filt_quant)
    print(mse)
    mses.append(mse)
plt.show()
foo()
# plt.semilogy(np.array(bit_range)+OPT_INT_BITS+1, mses)
# plt.ylim((-1,1000))
# plt.xlim(MIN_BITS, MAX_BITS)
# plt.xlabel("Data word length [bits]")
# plt.ylabel("Quantization noise power [LSB^2]")
# plt.title("Loop filter quantization noise versus data word length,\n Two's complement fixed point representation")
# plt.grid()
# razavify()
# plt.tight_layout()
# plt.savefig("lf_quant_noise.pdf")
# plt.axhline(var_ratio*ntdc_post_lf)
# plt.subplot(2,1,1)
# plt.plot(x)
# plt.subplot(2,1,2)
# plt.plot(filt_ideal)
# plt.plot(filt_quant)
# plt.show()
# foo()

""" Optimization for filter error
"""
plt.clf()

fmin = 1e2
points = 512

a = [lf_params["a0"], lf_params["a1"]]
b = [lf_params["b0"], lf_params["b1"], lf_params["b2"]]
f, h_ideal = scipy.signal.freqz(a, b, np.geomspace(fmin, FCLK/2, points), fs=FCLK)
s = 2j*np.pi*f
l = (TDC_STEPS/DIV_N)*KDCO*h_ideal/s
g = l/(1+l)
plt.semilogx(f, 20*np.log10(np.abs(g)), label="Ideal")
MIN_BITS = 8
bit_range = range(MIN_BITS-OPT_INT_BITS-1, MAX_BITS-OPT_INT_BITS)
mses = []
for frac_bits in bit_range:
    _lf_params = quant_lf_params(lf_params, OPT_INT_BITS, frac_bits)
    print(_lf_params)
    a = [_lf_params["a0"], _lf_params["a1"]]
    b = [_lf_params["b0"], _lf_params["b1"], _lf_params["b2"]]
    f, h = scipy.signal.freqz(a, b, np.geomspace(fmin, FCLK/2, points), fs=FCLK)
    s = 2j*np.pi*f
    l = (TDC_STEPS/DIV_N)*KDCO*h/s
    g = l/(1+l)
    # w, h = scipy.signal.freqz(a, b, points)
    plt.semilogx(f, 20*np.log10(np.abs(g)), label="%d bits"%(frac_bits+OPT_INT_BITS+1))
    mses.append(np.var(20*np.log10(np.abs(h[1:]))-20*np.log10(np.abs(h_ideal[1:]))))
# plt.legend()
# plt.grid()
# plt.xlim((1e3, 1e6))
# plt.ylim((-6, 3))
# plt.title("Digital filter response versus data word resolution,\n Two's complement fixed point representation")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude $\hat T(f)$ [dB]")
# razavify(loc="lower left", bbox_to_anchor=[0,0])
# plt.xticks([1e3, 1e4, 1e5, 1e6], ["$10^3$", "$10^4$", "$10^5$", "$10^6$"])
# # plt.show()
# plt.tight_layout()
# plt.savefig("tf_quant_error.pdf")
# plt.show()


plt.subplot(3,1,3)
plt.clf()
print(mses)
mses=np.array(mses)
P = 1
Z = 1
n = np.array(range(MIN_BITS-OPT_INT_BITS-1, MAX_BITS-OPT_INT_BITS))
compl = n*(P+Z) + (P+Z+1)*n**2

# foo()

plt.legend()
plt.grid()
plt.title("Filter mean squared error (MSE) versus data word resolution,\n Two's complement fixed point representation")
plt.xlabel("Data word length [bits]")
plt.ylabel("MSE [dB^2]")
plt.semilogy(n+OPT_INT_BITS+SIGN_BIT, mses)
# plt.plot(range(MIN_BITS-OPT_INT_BITS-1, MAX_BITS-OPT_INT_BITS), np.sqrt(mses)*compl)
plt.xlim((MIN_BITS, MAX_BITS))
# plt.ylim((-0.05, 1.2))
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"])
razavify()
plt.tight_layout()
plt.savefig("tf_mse.pdf")
plt.show()



