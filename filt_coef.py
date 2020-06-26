""" Gradient descent optimization using MSE PLL closed loop transfer funtion
    to ideal second order response
"""
import numpy as np
import matplotlib.pyplot as plt
from libpll.optimize import grad_descent
from libpll.plot import razavify
from copy import copy
import scipy.signal
import json
# open loop jitter transfer functions

def _pll_ojtf2(f, m, n, kdco, ki, fz, fp, delay):
    return -kdco*ki*(m/float(n))*(1j*f/fz+1)*np.exp(-2j*np.pi*f*delay)/((1j*f/fp+1)*(2*np.pi*f)**2)
pll_ojtf2 = np.vectorize(_pll_ojtf2, otypes=[complex])

def _pll_ojtf(f, k, fz, fp, delay):
    return -k*(1j*f/fz + 1)*np.exp(-2j*np.pi*f*delay)/((1j*f/fp+1)*(2*np.pi*f)**2)
pll_ojtf = np.vectorize(_pll_ojtf, otypes=[complex])

def _lpf_so(f, fn, damping):
    wn = 2*np.pi*fn
    w = 2*np.pi*f
    return wn**2/(-w**2 + 2j*damping*wn*w + wn**2)
lpf_so = np.vectorize(_lpf_so, otypes=[complex])

def _lf(f, ki, fz, fp):
    return ki*(1j*f/fz+1)/((1j*f/fp+1)*(2j*np.pi*f))
lf = np.vectorize(_lf, otypes=[complex])

def _tdc_noise(fref, n, g, tdel):
    return fref*np.abs(2*np.pi*n*g)**2*tdel**2/12
tdc_noise = np.vectorize(_tdc_noise, otypes=[float])

def _dco_noise(f0, df, power, temp):
    """ Returns the theorectical pn limit for ring oscillator
    """
    return 7.33*KB*temp*(f0/df)**2/power
dco_noise = np.vectorize(_dco_noise, otypes=[float])

def fixed_point(v, frac_bits):
    return int(round(v*(2**frac_bits)))

KB = 1.38064852e-23
FMAX = 1e7
STEPS = 100
INT_STEPS = 10000
M = 64
N = 150 #150
BW = 50e3
KDCO = 10e3
DAMPING = 0.707
DELAY = 0.0
FCLK = 16e6
T = 1.0/FCLK
POSC = 50e-6
TEMP = 293

df = FMAX/STEPS
FREQS = np.geomspace(1, FMAX, int(STEPS))

DSP_BITS = 8

# for damp in np.linspace(0.5, 1.5, 11):
    # plt.semilogx(FREQS, 20*np.log10(np.abs(1-lpf_so(FREQS, BW, damp))), label="%.1f"%damp)
# plt.grid()
# plt.legend()
# plt.show()
# foo()

def cost_lf(fn, damping, fmax, steps, delay=0.0):
    df = fmax/steps
    freqs = np.geomspace(1, fmax, int(steps))
    def f(k, fz, fp):
        a = pll_ojtf(f=freqs, k=k, fz=fz, fp=fp, delay=delay)
        g = a/(1+a)
        # plt.clf()
        # plt.plot(np.log10(np.abs(lpf_so(freqs, fn, damping))))
        # plt.plot(np.log10(np.abs(g)))
        # plt.show()
        # foo()
        return np.mean(20*np.log10(np.abs(lpf_so(freqs, fn, damping)))
                - 20*np.log10(np.abs(g)))**2

    return f

f = cost_lf(BW, DAMPING, FMAX, STEPS, DELAY)
_freqs = np.linspace(FMAX/INT_STEPS, FMAX, int(INT_STEPS))
if False:
    sweep_data = {}
    for k in np.geomspace(1e4, 1e10, 13):
        print("\n*k = %E"%k)
        init = dict(k=k, fz=-0.1*BW, fp=1.5*BW)
        # init = dict(k=3e9, fz=0.1*BW, fp=np.inf)

        opt = grad_descent(f, ("fz", "fp"), init, conv_tol=1e-5, deriv_step=1e-10)

        sweep_data[str(k)] = copy(opt)

        a = pll_ojtf(_freqs, delay=0, **opt)
        g = a/(1+a)

        ndco = dco_noise(FCLK*N, _freqs, POSC, TEMP)*np.abs(1-g)**2
        ntdc = tdc_noise(FCLK, N, g, T/float(M))

        sweep_data[str(k)]["int_ntdc"] = np.sum(ntdc)*df
        sweep_data[str(k)]["int_ndco"] = np.sum(ndco)*df

        for _k,v in sweep_data[str(k)].items():
            print("%s -> %E"%(_k,v))


    f = open("lf_gain_sweep2.dat", "w")
    f.write(json.dumps(sweep_data))
    f.close()

    ks = []
    p_ndco = []
    p_ntdc = []
    p_sum = []
    for k,v in sweep_data.items():
        ks.append(float(k))
        p_ndco.append(v["int_ndco"])
        p_ntdc.append(v["int_ntdc"])
        p_sum.append(v["int_ndco"]+v["int_ntdc"])

    plt.semilogx(ks, 10*np.log10(p_ndco), label="DCO")
    plt.semilogx(ks, 10*np.log10(p_ntdc), label="TDC")
    plt.semilogx(ks, 10*np.log10(p_sum), label="Combined")
    plt.title("OJTF K versus noise power")
    plt.xlabel("k")
    plt.ylabel("Power [dB$_{relative}$]")
    plt.grid()
    plt.legend()
    plt.show()

    foo()
#


init = dict(k=1e10, fz=0.1*BW, fp=1.5*BW)
# init = dict(k=3e9, fz=0.1*BW, fp=np.inf)

f = cost_lf(BW, DAMPING, FMAX, STEPS, DELAY)

opt = grad_descent(f, ("fz", "fp"), init, conv_tol=1e-5, deriv_step=1e-10)



_opt = copy(opt)
_opt["ki"] = opt["k"]*N/(M*KDCO)
_opt["kp"] = _opt["ki"]/(2*np.pi*opt["fz"])
ki = _opt["ki"]
wz = 2*np.pi*_opt["fz"]
wp = 2*np.pi*_opt["fp"]
print("\n* Computed gain coeficients, pole/zeros locations:")
for k,v in _opt.items():
    print("%s -> %E"%(k,v))

a0 = ki*(wp/wz)*T*(1+wz*T)/(1+wp*T)
a1 = -ki*(wp/wz)*T/(1+wp*T)
b0 = 1
b1 = -(2+wp*T)/(1+wp*T)
b2 = 1/(1+wp*T)


print("\n* Loop filter difference equation")
print("y[n] = %.10E*x[n] + %.10E*x[n-1] + %.10E*y[n-1] + %.10E*y[n-2]"%(a0,a1,-b1,-b2))
print("a0 = %.10E"%a0)
print("a1 = %.10E"%a1)
print("b1 = %.10E"%b1)
print("b2 = %.10E"%b2)
MAX_TAP = max(np.abs([a0,a1,b1,b2]))
INT_BITS = np.ceil(np.log2(MAX_TAP))
FRAC_BITS = DSP_BITS - INT_BITS

# a = pll_ojtf2(FREQS, M, N, KDCO, KP, FZ, FP, DELAY)
a = pll_ojtf(FREQS, delay=0, **opt)
g = a/(1+a)

plt.subplot(1,3,1)
plt.semilogx(FREQS, 20*np.log10(np.abs(lpf_so(FREQS, BW, DAMPING))), label="Ideal")
plt.semilogx(FREQS, 20*np.log10(np.abs(g)), label="G(f)")
plt.semilogx(FREQS, 20*np.log10(np.abs(1-g)), label="1-G(f)")
plt.legend()
plt.grid()
plt.title("Closed loop responses")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain [dB]")

plt.subplot(1,3,2)
ndco = dco_noise(FCLK*N, FREQS, POSC, TEMP)*np.abs(1-g)**2
ntdc = tdc_noise(FCLK, N, g, T/float(M))

plt.semilogx(FREQS, 10*np.log10(ntdc), label="TDC")
plt.semilogx(FREQS, 10*np.log10(ndco), label="DCO")
plt.semilogx(FREQS, 10*np.log10(ndco+ntdc), label="Combined")
plt.legend()
plt.grid()
plt.title("SSB Phase noise")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase noise [dBc]")

# razavify()
plt.subplot(1,3,3)

w, h = scipy.signal.freqz([a0, a1], [b0, b1, b2], fs=1/T)
_h = lf(w[1:], ki, opt["fz"], opt["fp"])
plt.title("Ideal versus discrete loop filter")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain [dB]")
plt.semilogx(w[1:], 20*np.log10(np.abs(h[1:])), label="Discretized")
plt.semilogx(w[1:], 20*np.log10(np.abs(_h)), label="Ideal")
plt.grid()
plt.legend()
plt.show()
