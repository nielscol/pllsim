# Compute delta-f which contains 99% of power for PLL with second order butterworth response

import numpy as np
import matplotlib.pyplot as plt
from libpll.plot import razavify, adjust_subplot_space
KB = 1.38064852e-23

def _butterworth(f, fn, damping):
    wn = 2*np.pi*fn
    w = 2*np.pi*f
    return wn**2/(-w**2 + 2j*damping*wn*w + wn**2)
butterworth = np.vectorize(_butterworth, otypes=[complex])

def _tdc_noise(fref, n, g, tdel):
    return fref*np.abs(2*np.pi*n*g)**2*tdel**2/12
tdc_noise = np.vectorize(_tdc_noise, otypes=[float])

def _dco_noise(f0, df, power, temp):
    """ Returns the theorectical pn limit for ring oscillator
    """
    return 7.33*KB*temp*(f0/df)**2/power
dco_noise = np.vectorize(_dco_noise, otypes=[float])

def max_tdel(pn0, fref, n):
    return np.sqrt(12*pn0/(fref*(2*np.pi*n)**2))

FMAX = 1e8
STEPS = 1e6
FN = 100e3
BER = 1e-2
df = FMAX/STEPS
DAMPING = np.sqrt(0.5)
FREQS = np.linspace(FMAX/STEPS, FMAX, int(STEPS))

f_bers = []
tdc_steps = []
FNS = np.geomspace(1e4,1e6, 11)

for FN in FNS:
    print("\n*", FN)
    g = butterworth(FREQS, FN, DAMPING)

    ndco = dco_noise(2.4e9, FREQS, 50e-6, 293)


    _max_tdel = max_tdel(ndco[0]*np.abs(1-g[0])**2, 16e6, 150)
    print(_max_tdel, 1.0/(16e6*_max_tdel))

    ntdc = tdc_noise(16e6, 150, g, _max_tdel)

    _ndco = ndco*np.abs(1-g)**2
    _ncomb = _ndco + ntdc

    np.where
    int_ntdc = np.cumsum(ntdc)*df
    int_ndco = np.cumsum(_ndco)*df
    int_ncomb = np.cumsum(_ncomb)*df
    f_ber = FREQS[np.where(1+2*int_ncomb > (1-2*BER)*(1+2*int_ncomb[-1]))[-1][0]]
    print("*** ", f_ber)
    tdc_steps.append(1.0/(16e6*_max_tdel))
    f_bers.append(f_ber)

plt.subplot(2,1,1)
plt.semilogx(FNS, f_bers, color="k")
plt.title("Offset from carrier containing 99% of power\n vs Closed loop PLL bandwidth")
plt.xlabel("Loop bandwidth [Hz]")
plt.ylabel("99% Power Offset [Hz]")
#plt.grid()
razavify()

plt.subplot(2,1,2)
plt.semilogx(FNS, tdc_steps, color="k")
plt.title("Minimum number of TDC steps required\n vs PLL closed loopbandwidth")
plt.xlabel("Loop bandwidth [Hz]")
plt.ylabel("Minimum TDC steps")
#plt.grid()
razavify()
adjust_subplot_space(2,1)
plt.show()
# plt.subplot(1,3,1)
# plt.semilogx(FREQS, 20*np.log10(np.abs(g)), label="TDC noise")
# plt.semilogx(FREQS, 20*np.log10(np.abs(1-g)**2), label="DCO noise")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude response [dB]")
# plt.title("TDC, DCO Noise Transfer functions")
# plt.grid()
# plt.xlim((1e2,FMAX))
# plt.legend()
# #plt.show()
# 
# plt.subplot(1,3,2)
# plt.semilogx(FREQS, 10*np.log10(ntdc), label="TDC")
# plt.semilogx(FREQS, 10*np.log10(_ndco), label="DCO")
# plt.semilogx(FREQS, 10*np.log10(_ncomb), label="Combined")
# plt.xlim((1e2,FMAX))
# 
# 
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Phase noise [dBc/Hz]")
# plt.title("TDC, DCO Phase noise contributions")
# plt.grid()
# plt.legend()
# 
# plt.subplot(1,3,3)
# plt.semilogx(FREQS, int_ntdc, label="TDC")
# plt.semilogx(FREQS, int_ndco, label="DCO")
# plt.semilogx(FREQS, int_ncomb, label="Combined")
# plt.xlim((1e2,FMAX))
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Phase noise power [dBc]")
# plt.title("TDC, DCO Integrated phase noise")
# plt.grid()
# plt.legend()



plt.show()
