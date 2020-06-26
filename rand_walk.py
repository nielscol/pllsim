""" Trying to figure out the whole discrete random walk thing to model vco phase noise
    in the time domain
"""

import numpy as np
import matplotlib.pyplot as plt
from libpll._signal import make_signal
from libpll.plot import plot_fd, plot_td, plot_pn_ssb, razavify
from libpll.pllcomp import *
from libpll.analysis import find_rw_k, meas_ref_spur, eval_model_pn, average_pn
from libpll.pllcomp import DCO

KB = 1.38064852e-23

def phase_noise_est(k, n, df, tstep, m=0.61):
    return k**2/(n*(2*np.pi*df*tstep*m)**2)

def min_pn_ro(f0, df, power, temp):
    """ Returns the theorectical pn limit for ring oscillator
    """
    return 7.33*KB*temp*(f0/df)**2/power

def min_power_ro(pn, fo, df, temp):
    """ Returns minimum possible needed to achieve a given phase noise value
    """
    return 7.33*KB*temp*(f0/df)**2/pn

def rw_gain(pn, df, n, tstep, m=0.61):
    """ calculates random walk gain to produce 1/f dependent phase noise
    args:
        pn : phase noise at offset df
        df : phase noise measurement offset from carrier
        n  : number of simulation samples
        tstep : time step of simulation
        m  : magic correction factor, used to match simulated values to theory
    """
    return 2*np.pi*df*tstep*m*np.sqrt(n*pn)

def ro_rw_model_param(f0, power, temp, n, tstep, m=0.61):
    """ Calculates model parameter for ring oscillator random walk phase model
        for the theoretical performance limit
    """
    return 2*np.pi*tstep*m*np.sqrt(7.33*KB*temp*n*f0/power)


SAMPLES = int(1e6)
# K = 1e-6
FS = 8e9
F0 = 8e8
PN_DF = 1e6

a = ro_rw_model_param(f0=F0, power=50e-6, temp=293, n=SAMPLES, tstep=1.0/FS)
print(10*np.log10(min_pn_ro(F0, PN_DF, 50e-6, 293)))
print(a*np.sqrt(F0))
print(a*np.sqrt(F0+100e6))

TARG_PN_DB = -94.2
TARG_PN = 10**(TARG_PN_DB/10.0)
K = rw_gain(TARG_PN, PN_DF, SAMPLES, 1.0/FS)
print("k=%E"%K)

# Random NRZ sequence
a = np.random.choice((-K,K), SAMPLES)
A = np.fft.fft(a)
_A = np.abs(A)
print("%E"%(np.sum(np.abs(A)**2)))

# Solid tone at FS
phase = np.arange(SAMPLES)*(1.0/FS)*2.0*np.pi*F0
b = np.cos(phase)
B = np.fft.fft(b)
_B = np.abs(B)

# Random walk based on summation of the NRZ seq
c = np.cumsum(a)
C = np.fft.fft(c)
_C = np.abs(C)

# Tone at FS with random walk added to phase
d = np.cos(phase + c)
D = np.fft.fft(d)
_D = np.abs(D)


# Use DCO class model

krwro = ro_rw_model_param(f0=F0, power=50e-6, temp=293, n=SAMPLES, tstep=1.0/FS)
dco = DCO(kdco=1, f0=F0, dt=1.0/FS, krwro=krwro)
e = np.zeros(SAMPLES)
for n in range(SAMPLES):
    e[n] = dco.update(fctrl=100e6)


#sig_c = make_signal(td=c, fs=FS)
sig_d = make_signal(td=d, fs=FS)
sig_e = make_signal(td=e, fs=FS)


plot_pn_ssb(sig_d, f0=F0, dfmax=10e6)
plot_pn_ssb(sig_e, f0=F0+100e6, dfmax=10e6)

#print(20*np.log10(eval_model_pn(sig_c, 0, PN_DF)))
meas = 20*np.log10(eval_model_pn(sig_d, F0, PN_DF))
equ = 10*np.log10(phase_noise_est(K, SAMPLES, PN_DF, 1.0/FS))
print("Meas=%.2f, theory=%.2f, delta=%.2f"%(meas, equ, meas-equ))
meas_dco = 20*np.log10(eval_model_pn(sig_e, F0+100e6, PN_DF))
print("Meas dco=%.2f"%meas_dco)
# plt.plot(20*np.log10(_A))
# plt.plot(20*np.log10(_B))
# plt.plot(20*np.log10(_C))
# plt.plot(20*np.log10(_D))

razavify()
plt.show()

