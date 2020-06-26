import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from copy import copy
from libpll.tools import timer, binary_lf_coefs
from libpll.pllcomp import ClockPhase, SCPDPhase, BBPD, DCOPhase, LoopFilterPIPhase
from libpll._signal import make_signal
from libpll.pncalc import min_ro_pn, fom_to_pn, rw_gain_fom
from libpll.opt_pll import design_filters, tsettle_lf, s0_osc
from libpll.opt_lf_bits import opt_lf_num_bits
from libpll.plot import plot_fd, plot_td, plot_pn_ssb2, razavify, adjust_subplot_space
from libpll.plot import plot_loop_filter, plot_lf_ideal_pn, plot_pn_ar_model
from libpll.analysis import pn_signal, meas_inst_freq, est_tsettle_pll_tf, noise_power_ar_model


FOSC = 816e6                # Oscillator frequency
FBIN = 1e3 #1e1 #1e1              # Target frequency bin size (for PN spectrum)
FS = FCLK = 16e6             # Clock reference frequency
USE_BBPD = True
BBPD_RMS_JIT = 1.4e-12

LF_MIN_BITS = 8
LF_MAX_BITS = 16
LF_RMS_FILT_ERROR = 0.1 # dB
LF_NOISE_FIGURE = 0.1 #dB - maximum increase in noise due to LF
ALPHA_MAX = 0.1     # max BW/FCLK
FORCE_INT_BITS = False
INT_BITS = 32
FORCE_FRAC_BITS = True
FRAC_BITS = 8


DCO_FINE_BITS = 10
DCO_MED_BITS = 3
KVCO1 = 5.378e6            # Fine DCO gain
KVCO2 = 3.092e7            # Medium DCO gain
VDD = 0.8
FL_DCO = 803.4e6          # starting frequency of DCO with fctrl=0
INIT_F_ERR = -1.0e6
SETTLE_DF_SCPD = 1e6
SETTLE_DF_BBPD = 1e5
MAX_TSETTLE = 50e-6
FORCE_GEAR_SW_T = False
GEAR_SW_T = 0e-6
GEAR_SW_SAFETY_FACTOR = 8 # use extra time before switching gears


DCO_POWER = 80e-6        # Calcuate PN based on RO physical limits
TEMP = 300                 # Simulation temperature, K
DCO_PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier
DCO_FOM = -160
USE_THER_DCO_PN = False    # Use the theoretical value for ring oscillator PN

PLOT_SPAN = 50e-6
PN_FMAX = 1e8           # Limit max frequency of PN plot

SIM_STEPS = int(1.6e4)
FBIN = FS/float(SIM_STEPS)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Determine DCO phase noise

if USE_THER_DCO_PN:
    DCO_PN = min_ro_pn(FOSC, DCO_PN_DF, DCO_POWER, TEMP)
    DCO_FOM = 10*np.log10(DCO_PN*FOSC**2*DCO_POWER*1e3/DCO_PN_DF**2)
else:
    DCO_PN = fom_to_pn(DCO_FOM, DCO_POWER, FOSC, DCO_PN_DF)
print("\nOscillator characteristics:")
print("\tFOM = %.1F dB"%DCO_FOM)
print("\tPower = %.1f uW"%(DCO_POWER*1e6))
print("\tTEMP = %.1f K"%TEMP)
print("\tL(df=%.2E) = %.2f dBc/Hz"%(DCO_PN_DF, 10*np.log10(DCO_PN)))

print(DCO_PN)
# KRWRO = ro_rw_model_param(f0=FOSC, power=DCO_POWER, temp=TEMP, n=SIM_STEPS, tstep=1.0/FS)
# krw2 = rw_gain(DCO_PN, DCO_PN_DF, SIM_STEPS, tstep=1/FS, m=1.0)
# print(KRWRO, krw2)

# a = np.random.choice((-KRWRO,+KRWRO), SIM_STEPS)
# b = np.random.choice((-krw2, +krw2), SIM_STEPS)
# c = np.cumsum(a)
# d = np.cumsum(b)

# C = np.fft.fft(c)*(1/SIM_STEPS)
# D = np.fft.fft(d)*(1/SIM_STEPS)


f = np.arange(1,int(0.5*SIM_STEPS))*(FCLK/SIM_STEPS)

# plt.semilogx(f, 20*np.log10(np.abs(C[1:int(0.5*SIM_STEPS)])))
# plt.semilogx(f, 20*np.log10(np.abs(D[1:int(0.5*SIM_STEPS)])))
# plt.grid()
# plt.show()
PN_DF = 1e6
s0 = s0_osc(DCO_FOM, DCO_POWER, FOSC)
# krw = 2*np.pi*np.sqrt(s0)/float(FS)
# krw = 4*np.pi*np.sqrt(FS*s0)/np.sqrt((2*FS)**2 + (2*np.pi*PN_DF)**2)
# krw = 2*np.pi*np.sqrt(s0/FS)
krw = rw_gain_fom(fom_db=DCO_FOM, fosc=FOSC, power=DCO_POWER, fs=FS)
print("krw =", krw)
error = np.zeros(SIM_STEPS)
dco1 = DCOPhase(kdco1=0, kdco2=0, f0=FL_DCO, dt=1/float(FS), krw=krw,
               init_phase=0, quantize=False)
dco2 = DCOPhase(kdco1=0, kdco2=0, f0=FL_DCO, dt=1/float(FS), krw=0,
               init_phase=0, quantize=False)
for n in range(SIM_STEPS):
    error[n] = dco1.update(0,0)-dco2.update(0,0)
# print(error)
# plt.plot(error)


E = np.fft.fft(error)/np.sqrt(FS*SIM_STEPS)
plt.semilogx(f, 20*np.log10(np.abs(E[1:int(0.5*SIM_STEPS)])))
# plt.show()
K = np.average(np.abs(E[1:int(0.5*SIM_STEPS)])**2*f**2)
print("Fitted pn @ 1MHz = ", 10*np.log10(K/1e6**2))
print("rms_error =", np.std(error))
print("predicted rms error =", np.sqrt(2*K*(1/FBIN - 1/(0.5*FS))))
# plt.plot(K)
# plt.show()

pn_sig = make_signal(error, fs=FS)
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=0)
plot_pn_ar_model(pn_sig, p=200, tmin=0)
plt.show()

