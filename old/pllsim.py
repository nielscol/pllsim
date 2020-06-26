""" Discrete time integer-N PLL simulation
    Cole Nielsen 2019
"""
# TODO
# - [ ] Lock detect/estimation
# - [ ] Interpolation 
# - [ ] Fix phase noise plot scaling
# - [ ] Gate signal in PN to avoid start-up transient
# - [ ] Quantized levels in loop filter
# - [ ] Pull-in/lock-in range analysis

import numpy as np
import matplotlib.pyplot as plt
from libpll._signal import make_signal
from libpll.plot import plot_fd, plot_td, plot_pn_ssb, razavify, adjust_subplot_space
from libpll.pllcomp import *
from libpll.analysis import find_rw_k
from libpll.pncalc import ro_rw_model_param, min_pn_ro
import time

###############################################################################
# Simulation parameters
###############################################################################

F0 = 2.4e9                # Oscillator frequency
MIN_FS = 10*F0          # 1/tstep of simulation
FBIN = 1e4              # Target frequency bin size (for PN spectrum)
FCLK = 16e6             # Clock reference frequency
# FN_LF = 5e4             # Natural frequency of loop filter
# DAMPING_LF = 0.707      # Damping of loop filter
DIV_N = 150              # divider modulus
TDC_STEPS = 64          # 


KDCO = 10e3              # DCO gain per fctrl word LSB
FL_DCO = 2.3e9          # starting frequency of DCO with fctrl=0
# LF_INIT = 16600         # Initial loop filter state (in=out=LF_INIT)
INIT_F_ERR = -15e6

# Phase noise 
# PN = -84.7                # dBc/Hz, target phase noise at PN_DF
RO_POWER = 50e-6        # Calcuate PN based on RO physical limits
T = 293                 # Simulation temperature, K
PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier

PLOT_SLICE = slice(1000000) # matplotlib is slow with too many points, reduce for time domain
PN_FMAX = 1e7           # Limit max frequency of PN plot

# y[n] = 1.4536353335E+00*x[n] + -1.4491224320E+00*x[n-1] + 1.9722727330E+00*y[n-1] + -9.7227273304E-01*y[n-2]
a0 = 1.3082718002E+00
a1 = -1.3042101888E+00
b1 = -1.9722727330E+00
b2 = 9.7227273304E-01


###############################################################################
# Simulation parameters
###############################################################################

# if COMPUTE_K:
    # K = find_rw_k(F0, FS, SAMPLES, -87, 1e6, SEED)

LF_IDEAL = int(round((F0-FL_DCO)/KDCO))
LF_INIT = int(round((F0-FL_DCO+INIT_F_ERR)/KDCO))

MIN_TCLK_STEPS = int(round(MIN_FS/FCLK)) # minimum steps per reference cycle needed
TCLK_STEPS = TDC_STEPS*int(np.ceil(MIN_TCLK_STEPS/float(TDC_STEPS))) # actual steps per reference cycle
FS = TCLK_STEPS*FCLK
SAMPLES = int(round(FS/FBIN)) # make simulation long enough to get desired fbin

TDC_OFFSET = int(round(TDC_STEPS/2.0)) # offset to get TDC zero difference to be mid-word


SS_FCTRL = (F0 - FL_DCO)/KDCO
TDC_SCALE = SS_FCTRL/float(TDC_OFFSET)

krwro = ro_rw_model_param(f0=F0, power=RO_POWER, temp=T, n=SAMPLES, tstep=1.0/FS)

print("\n* Simulation samples = %E Sa"%SAMPLES)
print("\t Fs = %E Hz"%FS)
print("\t Fbin = %E Hz"%FBIN)

###############################################################################
# Initialize PLL component objects, arrays to save data
###############################################################################

# PLL components from libpll.pllcomp
clk = Clock(FCLK, 1/FS)
tdc = TDC(TDC_STEPS, TCLK_STEPS)
dco = DCO(kdco=KDCO, f0=FL_DCO, dt=1.0/FS, krwro=krwro, quantize=True)
div = Divider()
lf = LoopFilterIIR(a0, a1, b1, b2, init=LF_INIT)
# lf = LoopFilterSecOrder(1/FCLK, FN_LF, DAMPING_LF, init=LF_INIT)


# data-save arrays
osc_out = np.zeros(SAMPLES)
clk_out = np.zeros(SAMPLES)
lf_out = np.zeros(SAMPLES)
div_out = np.zeros(SAMPLES)
tdc_out = np.zeros(SAMPLES)
# osc_out[0] = clk_out[0] = div_out[0] = tdc_out[0] = lf_out[0] = 0

##############################################################################
# Simulation loop
###############################################################################

t0 = time.clock()
for n in range(SAMPLES)[1:]:
    clk_out[n] = clk.update()
    tdc_out[n] = tdc.update(clk=clk_out[n-1], xin=div_out[n-1])
    # _tdc = TDC_SCALE*((TDC_OFFSET+tdc_out[n-1])%TDC_STEPS)
    _tdc = ((TDC_OFFSET+tdc_out[n-1])%TDC_STEPS)-TDC_OFFSET
    lf_out[n]  = lf.update(xin=_tdc, clk=clk_out[n-1])
    osc_out[n] = dco.update(lf_out[n-1])
    div_out[n] = div.update(osc_out[n-1], DIV_N)
tdelta = time.clock() - t0
print("\nSimulation completed in %f s"%tdelta)


###############################################################################
# Plot data
###############################################################################

osc_sig_full = make_signal(td=osc_out[len(osc_out)/2:], fs=FS)
osc_sig = make_signal(td=osc_out[PLOT_SLICE], fs=FS)
clk_sig = make_signal(td=clk_out[PLOT_SLICE], fs=FS)
lf_sig = make_signal(td=lf_out[PLOT_SLICE], fs=FS)
div_sig = make_signal(td=div_out[PLOT_SLICE], fs=FS)
tdc_sig = make_signal(td=tdc_out[PLOT_SLICE], fs=FS)

plt.subplot(2,3,1)
plot_td(clk_sig, title="CLK")
# razavify()
plt.subplot(2,3,2)
plot_td(osc_sig, title="DCO")
# razavify()
plt.subplot(2,3,3)
plot_td(div_sig, title="DIV")
# razavify()
plt.subplot(2,3,4)
plot_td(tdc_sig, title="TDC")
# razavify()
plt.subplot(2,3,5)
plot_td(lf_sig, title="LF")
# razavify()
plt.subplot(2,3,6)
plot_pn_ssb(osc_sig_full, F0, dfmax=PN_FMAX, line_fit=False)
# plot_fd(osc_sig_full)
# razavify()
adjust_subplot_space(2,3,0.1,0.1)
plt.show()

