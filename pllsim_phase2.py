""" PHASE DOMAIN Discrete time integer-N PLL simulation
    (has better resolution for phase noise)
    Cole Nielsen 2019
"""
# TODO
# - [ ] Lock detect/estimation
# - [ ] Interpolation 
# - [ ] Gate signal in PN to avoid start-up transient
# - [ ] Pull-in/lock-in range analysis

import numpy as np
import matplotlib.pyplot as plt
from libpll._signal import make_signal
from libpll.plot import plot_fd, plot_td, plot_pn_ssb2, razavify, adjust_subplot_space
from libpll.pllcomp import *
from libpll.engine import run_sim
from libpll.analysis import find_rw_k
from libpll.pncalc import ro_rw_model_param, min_pn_ro
from libpll.filter import calc_loop_filter, plot_loop_filter, plot_lf_ideal_pn
from libpll.tools import timer
import time
import json
###############################################################################
# Simulation parameters
###############################################################################

FOSC = 2.4e9                # Oscillator frequency
# MIN_FS = 10*FOSC          # 1/tstep of simulation
FBIN = 1e1              # Target frequency bin size (for PN spectrum)
FS = FCLK = 16e6             # Clock reference frequency
DIV_N = 150              # divider modulus
TDC_STEPS = 150           # 
BBPD_TSU = BBPD_TH = 10e-12

INT_BITS = 16 #= 4
FRAC_BITS = 12 #= 12

KDCO = 10e3             # DCO gain per fctrl word LSB
FL_DCO = 2.3e9          # starting frequency of DCO with fctrl=0
# LF_INIT = 16600         # Initial loop filter state (in=out=LF_INIT)
INIT_F_ERR = -24e6
K_OL = 3e10             # Open loop transfer function gain coefficient
LOOP_BW = 1e5           # PLL closed loop bandwidth
LOOP_DAMPING = 1.0      # PLL closed loop damping
USE_SAVED_LF = False     #
# Phase noise 
# PN = -84.7                # dBc/Hz, target phase noise at PN_DF
RO_POWER = 50e-6        # Calcuate PN based on RO physical limits
TEMP = 293                 # Simulation temperature, K
PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier

PLOT_SPAN = 200e-6
PN_FMAX = 1e7           # Limit max frequency of PN plot


###############################################################################
# Simulation parameters
###############################################################################

# if COMPUTE_K:
    # K = find_rw_k(FOSC, FS, SIM_STEPS, -87, 1e6, SEED)

KBBPD = fixed_point(TDC_STEPS*FCLK*(BBPD_TH+BBPD_TSU), INT_BITS, FRAC_BITS)
print("Kbb ->", KBBPD)

DT = 1/float(FS)
PLOT_SLICE = slice(0, int(PLOT_SPAN/DT), 1) # matplotlib is slow with too many points, reduce for time domain
LF_IDEAL = int(round((FOSC-FL_DCO)/KDCO))
LF_INIT = int(round((FOSC-FL_DCO+INIT_F_ERR)/KDCO))
print("LF_INIT=%f"%LF_INIT)
# MIN_TCLK_STEPS = int(round(MIN_FS/FCLK)) # minimum steps per reference cycle needed
# TCLK_STEPS = TDC_STEPS*int(np.ceil(MIN_TCLK_STEPS/float(TDC_STEPS))) # actual steps per reference cycle
# FS = TCLK_STEPS*FCLK
SIM_STEPS = int(round(FS/FBIN)) # make simulation long enough to get desired fbin

krwro = ro_rw_model_param(f0=FOSC, power=RO_POWER, temp=TEMP, n=SIM_STEPS, tstep=1.0/FS)
# 
print("\n* Simulation samples = %E Sa"%SIM_STEPS)
print("\t Fs = %E Hz"%FS)
print("\t Fbin = %E Hz"%FBIN)
if USE_SAVED_LF:
    f = open("lf_params.json", "r")
    lf_params = json.loads(f.read())
    f.close()
else:
    lf_params = calc_loop_filter(K_OL, LOOP_BW, LOOP_DAMPING, TDC_STEPS, DIV_N, KDCO, FCLK, delay=DT)
    f = open("lf_params.json", "w")
    f.write(json.dumps(lf_params))
    f.close()

plt.figure(1)
plot_loop_filter(RO_POWER, TEMP, **lf_params)
###############################################################################
# Initialize PLL component objects, arrays to save data
###############################################################################

# PLL components from libpll.pllcomp
clk = ClockPhase(FCLK, DT)
tdc = TDCPhase(TDC_STEPS)
bbpd = BBPD(BBPD_TSU, BBPD_TH, FCLK)
dco = DCOPhase(kdco=KDCO, f0=FL_DCO, dt=DT, krwro=krwro, quantize=True)
lf = LoopFilterIIRPhase(init=LF_INIT, int_bits=INT_BITS, frac_bits=FRAC_BITS, **lf_params)

init_params = {
    "osc"   : 0,
    "clk"   : 0,
    "lf"    : LF_INIT,
    "div"   : 0,
    "tdc"   : 0,
    "bbpd"  : 0,
    "error" : 0,
}

def eval_step(n, step):
    step["osc"]   = dco.update(step["lf"])
    step["div"]   = step["osc"]/float(DIV_N)
    step["clk"]   = clk.update(n)
    step["tdc"]   = tdc.update(clk=step["clk"], xin=step["div"])
    step["bbpd"]  = bbpd.update(clk=step["clk"], xin=step["div"])
    step["error"] = step["tdc"] + KBBPD*step["bbpd"]
    step["lf"]    = lf.update(xin=step["error"], clk=step["clk"])
    return step

# data-save arrays
# osc_out = np.zeros(SIM_STEPS)
# clk_out = np.zeros(SIM_STEPS)
# lf_out = np.zeros(SIM_STEPS)
# div_out = np.zeros(SIM_STEPS)
# tdc_out = np.zeros(SIM_STEPS)
# bbpd_out = np.zeros(SIM_STEPS)
# error = np.zeros(SIM_STEPS)
# osc_out[0] = clk_out[0] = div_out[0] = tdc_out[0] = 0
# lf_out[0] = LF_INIT

##############################################################################
# Simulation loop
###############################################################################


sim_data = run_sim(eval_step, SIM_STEPS, init_params)


# from libpll.tools import is_edge_phase
# edge = np.zeros(SIM_STEPS)
# t0 = time.clock()
# for n in range(SIM_STEPS)[1:]:
    # osc_out[n]  = dco.update(lf_out[n-1])
    # div_out[n]  = osc_out[n]/float(DIV_N)
    # clk_out[n]  = clk.update(n)
    # edge[n] = is_edge_phase(clk_out[n], clk_out[n-1])
    # tdc_out[n]  = tdc.update(clk=clk_out[n], xin=div_out[n])
    # bbpd_out[n] = bbpd.update(clk=clk_out[n], xin=div_out[n])
    # error[n]    = tdc_out[n] + KBBPD*bbpd_out[n]
    # lf_out[n]   = lf.update(xin=error[n], clk=clk_out[n])
# tdelta = time.clock() - t0
# print("\nSimulation completed in %f s"%tdelta)

# plt.figure(2)
# plt.subplot(2,3,1)
# plt.plot(osc_out,)
# plt.title("DCO")
# plt.subplot(2,3,2)
# plt.plot(div_out,)
# plt.title("DIV")
# plt.subplot(2,3,3)
# plt.plot(np.diff(clk_out))
# plt.plot(edge)
# plt.title("CLK")
# plt.subplot(2,3,4)
# plt.plot(tdc_out,)
# plt.title("TDC")
# plt.subplot(2,3,5)
# plt.plot(lf_out,)
# plt.title("LF")
# plt.subplot(2,3,6)
# plt.plot(np.diff(osc_out)/(2*np.pi*DT))
# plt.show()

###############################################################################
# Plot data
###############################################################################


# phase_noise = osc_out-DIV_N*clk_out
# pn_sig_full = make_signal(td=phase_noise[int(len(phase_noise)/2):], fs=FS)
# pn_sig = make_signal(td=phase_noise[PLOT_SLICE], fs=FS)

# osc_sig_full = make_signal(td=osc_out[len(osc_out)/2:], fs=FS)
# osc_freq = make_signal(td=np.diff(osc_out[PLOT_SLICE])/(2*np.pi*DT), fs=FS)
# clk_sig = make_signal(td=clk_out[PLOT_SLICE], fs=FS)
# lf_sig = make_signal(td=lf_out[PLOT_SLICE], fs=FS)
# div_sig = make_signal(td=div_out[PLOT_SLICE], fs=FS)
# tdc_sig = make_signal(td=tdc_out[PLOT_SLICE], fs=FS)
# bbpd_sig = make_signal(td=bbpd_out[PLOT_SLICE], fs=FS)
# error_sig = make_signal(td=error[PLOT_SLICE], fs=FS)

phase_noise = sim_data["osc"]-DIV_N*sim_data["clk"]
pn_sig_full = make_signal(td=phase_noise[int(len(phase_noise)/2):], fs=FS)
pn_sig = make_signal(td=phase_noise[PLOT_SLICE], fs=FS)

osc_freq = make_signal(td=np.diff(sim_data["osc"][PLOT_SLICE])/(2*np.pi*DT), fs=FS)
lf_sig = make_signal(td=sim_data["lf"][PLOT_SLICE], fs=FS)
tdc_sig = make_signal(td=sim_data["tdc"][PLOT_SLICE], fs=FS)
bbpd_sig = make_signal(td=sim_data["bbpd"][PLOT_SLICE], fs=FS)
error_sig = make_signal(td=sim_data["error"][PLOT_SLICE], fs=FS)

# plt.subplot(2,3,1)
# plot_td(clk_sig, title="CLK")
# razavify()
plt.figure(3)
plt.subplot(2,3,1)
plot_td(osc_freq, title="Inst. DCO Frequency")
# razavify()
plt.subplot(2,3,2)
plot_td(pn_sig, title="Phase error (noise)")
# razavify()
plt.subplot(2,3,3)
plot_td(tdc_sig, title="TDC/PD Output", label="TDC")
plot_td(bbpd_sig, title="BBPD Output", label="BBPD")
plot_td(error_sig, title="Error Output", label="Combined Error")
plt.legend()
# razavify()
plt.subplot(2,3,4)
plot_td(lf_sig, title="Loop Filter Output")
# razavify()
plt.subplot(2,3,5)
plot_pn_ssb2(pn_sig_full, dfmax=PN_FMAX, line_fit=False)
plot_lf_ideal_pn(RO_POWER, TEMP, **lf_params)

plt.grid()
# plot_fd(osc_sig_full)
# razavify()
adjust_subplot_space(2,2,0.1,0.1)
plt.show()

