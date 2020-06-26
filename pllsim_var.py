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
import scipy.special
from libpll._signal import make_signal
from libpll.plot import plot_fd, plot_td, plot_pn_ssb2, razavify, adjust_subplot_space
from libpll.plot import plot_loop_filter, plot_lf_ideal_pn, plot_pllsim_lf_inst_freq
from libpll.plot import plot_pllsim_osc_inst_freq
# from libpll.pllcomp import *
from libpll.engine import run_sim, pllsim_int_n, sim_mc, pllsim_int_n_mp, sim_sweep
from libpll.pncalc import ro_rw_model_param, min_ro_pn
from libpll.filter import opt_pll_tf_so_type2, lf_from_pll_tf, opt_pll_tf_pi_controller
from libpll.filter import opt_lf_num_bits, opt_pll_tf_pi_controller_bbpd
from libpll.filter import opt_pll_tf_pi_controller_fast_settling
from libpll.tools import timer, fixed_point, binary_lf_coefs
from libpll.analysis import pn_signal, meas_inst_freq, est_tsettle_pll_tf
from libpll.analysis import meas_tsettle_pllsim, vector_meas, meas_pn_power_pllsim
import time
import json
###############################################################################
# Simulation parameters
###############################################################################

FOSC = 2.4e9                # Oscillator frequency
# FBIN = 1e3              # Target frequency bin size (for PN spectrum)
TMAX = 100e-6
FS = FCLK = 16e6             # Clock reference frequency
DIV_N = 150              # divider modulus
TDC_STEPS = 150           # 
BBPD_TSU = BBPD_TH = 2e-12 #2e-12

USE_BBPD = True

LF_MIN_BITS = 8
LF_MAX_BITS = 16
LF_RMS_FILT_ERROR = 0.1
LF_NOISE_FIGURE = 1 #dB - maximum increase in noise due to LF
# INT_BITS = 16 #= 4
# FRAC_BITS = 16 #= 12

KDCO = 10e3             # DCO gain per OTW LSB
FL_DCO = 2.3e9          # starting frequency of DCO with fctrl=0
# LF_INIT = 16600         # Initial loop filter state (in=out=LF_INIT)
INIT_F_ERR = -12e6
SETTLE_DF = 10e4
MAX_TSETTLE = 50e-6
SETTLE_DF_TDC = 10e4
SETTLE_DF_BBPD = 10e3
# K_OL = 1e10             # Open loop transfer function gain coefficient
INIT_LF_MODE_TDC = "speed" # "speed" or "phase_noise"
PHASE_MARGIN = 60
TSETTLE_SAFETY_FACTOR = 2.0

# LOOP_BW = 1e5           # PLL closed loop bandwidth
# LOOP_DAMPING = 0.5      # PLL closed loop damping
#USE_SAVED_LF = False     #
# Phase noise 
# PN = -84.7                # dBc/Hz, target phase noise at PN_DF
# RO_POWER = 50e-6        # Calcuate PN based on RO physical limits
TEMP = 293                 # Simulation temperature, K
PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier
DCO_POWER = 50e-6
DCO_PN_DF=1e6
DCO_POWER=50e-6
CHANNEL_BW = 1e6

PLOT_SPAN = 200e-6
PN_FMAX = 1e8           # Limit max frequency of PN plot

N_REF_SPURS = 6
FBIN_SPUR_SIM = 1e3
RUN_SPURSIM = False

N_MC_SAMPLES = 1000

OPT_KBBPD = False
OPT_FMAX = FCLK/2 # 5e6 #
MAX_ITER_BBPD_OPT = 5


###############################################################################
# Simulation parameters
###############################################################################



print("TDC_STEPS =", TDC_STEPS)

PLOT_SLICE = slice(0, int(PLOT_SPAN*FS), 1) # matplotlib is slow with too many points, reduce for time domain
LF_IDEAL = int(round((FOSC-FL_DCO)/KDCO))
LF_INIT = int(round((FOSC-FL_DCO+INIT_F_ERR)/KDCO))
print("* LF_INIT = %f"%LF_INIT)

# 
# if USE_SAVED_LF:
    # print("\n* Using saved loop filter coefficients")
    # f = open("lf_params.json", "r")
    # lf_params = json.loads(f.read())
    # f.close()
# else:
    # lf_params = calc_loop_filter(K_OL, LOOP_BW, LOOP_DAMPING,
                                 # TDC_STEPS, DIV_N, KDCO, FCLK, delay=1/float(FCLK))
    # f = open("lf_params.json", "w")
    # f.write(json.dumps(lf_params))
    # f.close()

# pll_tf = opt_pll_tf_so_type2(LOOP_BW, LOOP_DAMPING)
DCO_PN = min_ro_pn(FOSC, DCO_PN_DF, DCO_POWER, TEMP)
print("\nOscillator characteristics:")
print("\tPower = %E, TEMP = %f"%(DCO_POWER, TEMP))
print("\tL(df=%f) = %.2f dBc"%(DCO_PN_DF, 10*np.log10(DCO_PN)))

#######################################################################################
# Optimize loop filter for TDC mode
#######################################################################################
print("\n*** Loop filter optimization under TDC-based feedback")
if INIT_LF_MODE_TDC is "speed": # optimize for settling speed
    pll_tf_params = opt_pll_tf_pi_controller_fast_settling(PHASE_MARGIN, MAX_TSETTLE,
                                                           abs(SETTLE_DF_TDC/INIT_F_ERR),
                                                           FCLK, oversamp=30)
else:   # optimize for phase noise
    pll_tf_params = opt_pll_tf_pi_controller(MAX_TSETTLE, abs(SETTLE_DF_TDC/INIT_F_ERR), DCO_PN, DCO_PN_DF, TDC_STEPS,
                                      DIV_N, KDCO, FCLK, OPT_FMAX)
lf_params = lf_from_pll_tf(pll_tf_params, TDC_STEPS, DIV_N, KDCO, FCLK)

TSETTLE_EST = est_tsettle_pll_tf(pll_tf_params, SETTLE_DF_TDC/INIT_F_ERR)
print("\n* Estimated settling time of PLL (TDC mode) = %E s"%TSETTLE_EST)
print("\tBased on inital df=%E Hz, and a tolerance band of +/- %E Hz"%(INIT_F_ERR, SETTLE_DF_TDC))

_int_bits,_frac_bits = opt_lf_num_bits(lf_params, LF_MIN_BITS, LF_MAX_BITS, rms_filt_error=LF_RMS_FILT_ERROR, noise_figure=LF_NOISE_FIGURE, mode="tdc")
INT_BITS = 32
FRAC_BITS = _frac_bits


#######################################################################################
# Optimize loop filter for BBPD mode
#######################################################################################


if USE_BBPD:
    print("#################")
    if OPT_KBBPD:
        # KBBPD = opt_kbbpd(lf_params, DCO_PN, DCO_PN_DF, DCO_POWER, TEMP, TDC_STEPS, DIV_N, KDCO, FCLK,
                          # OPT_FMAX, BBPD_TSU, BBPD_TH, INT_BITS, FRAC_BITS, sim_steps=50000,
                          # max_iter=15)
        pll_tf_params_bbpd, sigma_ph = opt_pll_tf_pi_controller_bbpd(MAX_TSETTLE, abs(SETTLE_DF_BBPD/SETTLE_DF_TDC),
                                                           DCO_PN, DCO_PN_DF, DIV_N, KDCO, FCLK,
                                                           OPT_FMAX, max_iter=MAX_ITER_BBPD_OPT,
                                                                    delay=1/FCLK)
        lf_params_bbpd = lf_from_pll_tf(pll_tf_params_bbpd, 2*np.pi, DIV_N, KDCO, FCLK)
        k = np.sqrt(np.pi/2)*sigma_ph # linear gain of bbpd
        lf_params_bbpd["a0"] = k*lf_params_bbpd["a0"]
        lf_params_bbpd["a1"] = k*lf_params_bbpd["a1"]
        KBBPD = 1
    else:
        print("^^^^^^^^^^^^^^^^^^^^^^")
        KBBPD = np.sqrt(np.pi/2)*(0.5*TDC_STEPS*FCLK*(BBPD_TH+BBPD_TSU)) # sqrt(pi/2)*sigma_pn, pn estimated from setup/hold time
        print(INT_BITS, FRAC_BITS, 0.5*TDC_STEPS*FCLK*(BBPD_TH+BBPD_TSU))
        if INIT_LF_MODE_TDC is "speed":
            pll_tf_params_bbpd = opt_pll_tf_pi_controller(MAX_TSETTLE, abs(SETTLE_DF_TDC/INIT_F_ERR), DCO_PN, DCO_PN_DF, TDC_STEPS,
                                      DIV_N, KDCO, FCLK, OPT_FMAX)
            lf_params_bbpd = lf_from_pll_tf(pll_tf_params_bbpd, TDC_STEPS, DIV_N, KDCO, FCLK)
        else:
            pll_tf_params_bbpd = pll_tf_params
            lf_params_bbpd = lf_params
    #KBBPD = 1/np.sqrt(12*(1-2/np.pi))
    # KBBPD = 0.05
    TSETTLE_EST_BBPD = est_tsettle_pll_tf(pll_tf_params_bbpd, SETTLE_DF_BBPD/SETTLE_DF_TDC)
    print("\n* Estimated settling time of PLL (TDC mode) = %E s"%TSETTLE_EST_BBPD)
    print("\tBased on inital df=%E Hz, and a tolerance band of +/- %E Hz"%(SETTLE_DF_TDC, SETTLE_DF_BBPD))
    if OPT_KBBPD:
        _int_bits_bbpd,_frac_bits_bbpd = opt_lf_num_bits(lf_params_bbpd, LF_MIN_BITS, LF_MAX_BITS, rms_filt_error=LF_RMS_FILT_ERROR, noise_figure=LF_NOISE_FIGURE, mode="bbpd")
        FRAC_BITS = max(_frac_bits, _frac_bits_bbpd)
        _int_bits = max(_int_bits, _int_bits_bbpd)

    #make sure kbbpd != 0 when quantized
    print("* Kbb = ", KBBPD)
    while fixed_point(KBBPD, INT_BITS, FRAC_BITS) == 0.0:
        print("KBBPD is quantized to zero with current data word resolution, incrementing fractional bits by 1.")
        FRAC_BITS += 1
    KBBPD = fixed_point(KBBPD, INT_BITS, FRAC_BITS)
    print("* Quantized Kbb = ", KBBPD)
# M_BBPD = 1/(FCLK*(BBPD_TH+BBPD_TSU))
# else: KBBPD = 0
#TDC_STEPS = int(TDC_STEPS/float(KBBPD))

print("\n* Final resolution of data words: %d frac bits, %d int bits, 1 sign bit"%(FRAC_BITS, _int_bits))
print("\n* Digitized ilter coeffients in TDC mode")
binary_lf_coefs(lf_params, _int_bits, FRAC_BITS)
if USE_BBPD:
    print("\n* Digitized filter coeffients in BBPD mode")
    binary_lf_coefs(lf_params_bbpd, _int_bits, FRAC_BITS)

if USE_BBPD: EST_TOTAL_LOCK_TIME = TSETTLE_EST + TSETTLE_EST_BBPD
else: EST_TOTAL_LOCK_TIME = TSETTLE_EST
print("\n* Final estimated settling time of PLL (TDC and or BBPD mode) = %E s"%EST_TOTAL_LOCK_TIME)



###############################################################################
# Main phase noise simulation
###############################################################################

print("\n***********************************************")
print("* Running main phase noise simulation         *")
print("***********************************************")
# SIM_STEPS = int(round(FS/FBIN)) # make simulation long enough to get desired fbin
SIM_STEPS = int(TMAX*FCLK)
print("\n* Simulation samples = %E Sa"%SIM_STEPS)
print("\t Fs = %E Hz"%FS)
# print("\t Fbin = %E Hz"%FBIN)

KRWRO = ro_rw_model_param(f0=FOSC, power=DCO_POWER, temp=TEMP, n=SIM_STEPS, tstep=1.0/FS)
# INIT_F_ERR = 1.2e6
sim_params = {
    "fosc"        : FOSC,
    # "init_f_err"  : INIT_F_ERR,
    "init_f"      : FOSC + INIT_F_ERR,
    "fl_dco"      : FL_DCO,
    "fclk"        : FCLK,
    "fs"          : FS,
    "sim_steps"   : SIM_STEPS,
    "div_n"       : DIV_N,
    "tdc_steps"   : TDC_STEPS,
    "use_bbpd"    : USE_BBPD,
    "kbbpd"       : KBBPD,
    "bbpd_tsu"    : BBPD_TSU,
    "bbpd_th"     : BBPD_TH,
    "kdco"        : KDCO,
    "fl_dco"      : FL_DCO,
    "krwro_dco"   : KRWRO,
    "lf_i_bits"   : INT_BITS,
    "lf_f_bits"   : FRAC_BITS,
    "lf_params"   : lf_params,
    "lf_params_bbpd"   : lf_params_bbpd,
    "tsettle_est" : TSETTLE_EST*TSETTLE_SAFETY_FACTOR,
    "init_params" : {
        "osc"   : 0,
        "clk"   : 0,
        "lf"    : LF_INIT,
        "div"   : 0,
        "tdc"   : 0,
        "bbpd"  : 1,
        "error" : 0,
        "kbbpd" : 0.25,
    },
}

SAMPLE_CORRECTION = -np.sqrt(2)*scipy.special.erfcinv(2*(1-1.0/SIM_STEPS))
print("Sample size correction factor for df = %f"%SAMPLE_CORRECTION)

if False:
    sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "kdco", np.linspace(1e3, 20e3, 100))
    # sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "init_f", FOSC + np.linspace(-150e6, 150e6, 200))
    t_settle = vector_meas(meas_pn_power_pllsim, sweep_data, {"div_n":DIV_N, "tmin":TSETTLE_EST*TSETTLE_SAFETY_FACTOR})
    # t_settle = [x for x in t_settle if not np.isnan(x)]
    # for result in sweep_data:
        # plt.subplot(1,3,1)
        # plot_pllsim_osc_inst_freq(result)
        # plt.subplot(1,3,2)
        # plot_td(result["lf"])
    # plt.subplot(1,3,1)
    # plt.title("PLL instantaneous frequency")
    # plt.grid()
    # plt.subplot(1,3,2)
    # plt.title("Loop filter output (OTW)")
    # plt.grid()
    # plt.subplot(1,3,3)
    plt.plot(np.linspace(1e3, 20e3, 100), t_settle)
    # plt.plot(np.linspace(-60, 60, 200), t_settle)
    # plt.plot(np.linspace(-150, 150, 200), t_settle)
    plt.title("Phase noise variance vs $K_{DCO}$")
    plt.xlabel("$K_{DCO}$ [Hz/LSB]")
    plt.ylabel("Phase noise variance [rad^2]")
    # plt.xticks([-150, -100, -50, 0, 50, 100, 150])
    # plt.xticks([-60, -40, -20, 0, 20, 40, 60])
    # plt.ylim([0, 2e-5])
    plt.xlim((0,20000))
    plt.grid()
    razavify()
    # plt.yticks([0, 1e-5, 2e-5, 3e-5, 4e-5], ["0", "10", "20", "30", "40"])
    # plt.yticks([0, 5e-6, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    plt.xticks([0, 5000, 10000, 15000, 20000], ["0", "5000", "10000", "15000", "20000"])
    # foo()
    plt.tight_layout()
    plt.savefig("kdco_sweep_pn_var.pdf")
    plt.show()
    foo()

if False:
    sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "kdco", np.linspace(1e3, 30e3, 100))
    # sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "init_f", FOSC + np.linspace(-150e6, 150e6, 200))
    t_settle = vector_meas(meas_tsettle_pllsim, sweep_data, {"tol_hz":SETTLE_DF*SAMPLE_CORRECTION})
    # t_settle = [x for x in t_settle if not np.isnan(x)]
    # for result in sweep_data:
        # plt.subplot(1,3,1)
        # plot_pllsim_osc_inst_freq(result)
        # plt.subplot(1,3,2)
        # plot_td(result["lf"])
    # plt.subplot(1,3,1)
    # plt.title("PLL instantaneous frequency")
    # plt.grid()
    # plt.subplot(1,3,2)
    # plt.title("Loop filter output (OTW)")
    # plt.grid()
    # plt.subplot(1,3,3)
    plt.plot(np.linspace(1e3, 30e3, 100), t_settle)
    # plt.plot(np.linspace(-60, 60, 200), t_settle)
    # plt.plot(np.linspace(-150, 150, 200), t_settle)
    plt.title("PLL lock time vs $K_{DCO}$")
    plt.xlabel("$K_{DCO}$ [Hz/LSB]")
    plt.ylabel("Lock time [$\mu$s]")
    # plt.xticks([-150, -100, -50, 0, 50, 100, 150])
    # plt.xticks([-60, -40, -20, 0, 20, 40, 60])
    # plt.ylim([0e-5, 2e-5])
    # plt.ylim([1e-5, 5e-5])
    plt.xlim((0,30000))
    plt.grid()
    # razavify()
    # plt.yticks([0e-5, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "50"])
    # plt.yticks([1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["10", "20", "30", "40", "50"])
    # plt.yticks([0, 5e-6, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000], ["0", "5000", "10000", "15000", "20000", "25000", "30000"])
    # foo()
    plt.tight_layout()
    plt.savefig("__kdco_sweep.pdf")
    plt.show()
    foo()
if False:
    # SAMPLE_CORRECTION = 1
    # sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "kdco", np.linspace(2e3, 18e3, 100))
    # sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "init_f", FOSC + np.linspace(-100e6, 100e6, 200))
    sweep_data = sim_sweep(pllsim_int_n_mp, sim_params, "init_f", FOSC + np.linspace(-5e6, 5e6, 200))
    t_settle = vector_meas(meas_tsettle_pllsim, sweep_data, {"tol_hz":SETTLE_DF*SAMPLE_CORRECTION})
    # t_settle = [x for x in t_settle if not np.isnan(x)]
    for result in sweep_data:
        plt.subplot(1,3,1)
        plot_pllsim_osc_inst_freq(result)
        plt.subplot(1,3,2)
        plot_td(result["lf"])
    plt.subplot(1,3,1)
    plt.title("PLL instantaneous frequency")
    plt.grid()
    plt.subplot(1,3,2)
    plt.title("Loop filter output (OTW)")
    plt.grid()
    plt.show()
    foo()
    # plt.plot(np.linspace(2e3, 18e3, 100), t_settle)
    # plt.plot(np.linspace(-60, 60, 200), t_settle)
    # plt.plot(np.linspace(-100, 100, 200), t_settle)
    plt.plot(np.linspace(-5, 5, 200), t_settle)
    plt.title("PLL lock time vs initial frequency error")
    plt.xlabel("Initial frequency error [MHz]")
    plt.ylabel("Lock time [$\mu$s]")
    # plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100,])
    # plt.xticks([-60, -40, -20, 0, 20, 40, 60])
    # plt.ylim([0e-5, 4e-5])
    # plt.xlim((-100,100))
    plt.grid()
    razavify()
    # plt.yticks([0, 1e-5, 2e-5, 3e-5, 4e-5], ["0", "10", "20", "30", "40"])
    # plt.yticks([0, 5e-6, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    # plt.yticks([0.0, 1e-5, 2e-5, 3e-5, 4e-5], ["0", "10", "20", "30", "40"])
    # foo()
    plt.tight_layout()
    plt.savefig("__finit_sweep.pdf")
    plt.show()
    foo()


mc_data = sim_mc(pllsim_int_n_mp, sim_params, ["kdco", "init_f"], [0.2, 0.01], N_MC_SAMPLES)
# mc_data = sim_mc(pllsim_int_n_mp, sim_params, "kdco", 0.2, N_MC_SAMPLES)
t_settle = vector_meas(meas_tsettle_pllsim, mc_data, {"tol_hz":SETTLE_DF*SAMPLE_CORRECTION})
t_settle = [x for x in t_settle if not np.isnan(x)]

for result in mc_data:
    plt.subplot(1,3,1)
    plot_pllsim_osc_inst_freq(result)
    plt.subplot(1,3,2)
    plot_td(result["lf"])
plt.subplot(1,3,1)
plt.title("PLL instantaneous frequency")
plt.grid()
plt.subplot(1,3,2)
plt.title("Loop filter output (OTW)")
plt.grid()

plt.subplot(1,3,3)
plt.hist(np.array(t_settle)*1e6, bins=50)
plt.title("Lock time")
plt.xlabel("Time [$\mu$s]")
plt.ylabel("Count")
mean = np.mean(t_settle)
stdev = np.std(t_settle)
t_settle.sort()
tsettle_99 = t_settle[int(len(t_settle)*0.99)]
print("\nMonte Carlo simulation, Samples = %d"%N_MC_SAMPLES)
print("\tMean = %E"%mean)
print("\tStdev = %E"%stdev)
print("\tTsettle, 99%% CI = %E"%tsettle_99)
plt.show()
# foo()
# main_pn_data = pllsim_int_n(**main_sim_params)

plt.clf()
if True:
    for result in mc_data:
        # plt.subplot(1,3,1)
        plot_pllsim_osc_inst_freq(result)
        # plt.subplot(1,3,2)
        # plot_td(result["lf"])
    plt.title("PLL instantaneous frequency, Monte-Carlo Sampling, N=1000\n\
            # gear-switch at %.1fx estimated lock time of filter 1"%TSETTLE_SAFETY_FACTOR)
    plt.xlim((0,20e-6))
    # plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5], ["0", "10", "20", "30", "40"])
    plt.xticks([0, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    plt.ylabel("Frequency [GHz]")
    plt.xlabel("Time [$\mu$s]")
    razavify()
    plt.yticks([2.25e9, 2.3e9, 2.35e9, 2.4e9, 2.45e9, 2.5e9], ["2.25","2.30","2.35","2.40","2.45","2.50"])
    plt.tight_layout()
    plt.savefig("mc_trans_2x.pdf")
    plt.show()

plt.clf()

if False:
    plt.hist(np.array(t_settle)*1e6, bins=100)
    plt.title("Lock time Histogram, N=%d"%(N_MC_SAMPLES))
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Count")
    plt.grid()
    razavify()
    plt.tight_layout()
    plt.savefig("mc_hist.pdf")
    plt.show()
    foo()


###############################################################################
# Spur simulation
###############################################################################

if RUN_SPURSIM:

    print("\n***********************************************")
    print("* Running spur simulation                     *")
    print("***********************************************")

    FS_SPUR_SIM = 2*N_REF_SPURS*FCLK
    SPUR_SIM_STEPS = int(round(FS_SPUR_SIM/FBIN_SPUR_SIM))

    print("\n* Simulation samples = %E Sa"%SPUR_SIM_STEPS)
    print("\t Fs = %E Hz"%FS_SPUR_SIM)
    print("\t Fbin = %E Hz"%FBIN_SPUR_SIM)

    KRWRO = ro_rw_model_param(f0=FOSC, power=DCO_POWER, temp=TEMP, n=SPUR_SIM_STEPS,
                              tstep=1/float(FS_SPUR_SIM))

    spur_sim_params = {
        "fclk"        : FCLK,
        "fs"          : FS_SPUR_SIM,
        "sim_steps"   : SPUR_SIM_STEPS,
        "div_n"       : DIV_N,
        "tdc_steps"   : TDC_STEPS,
        "kbbpd"       : KBBPD,
        "bbpd_tsu"    : BBPD_TSU,
        "bbpd_th"     : BBPD_TH,
        "kdco"        : KDCO,
        "fl_dco"      : FL_DCO,
        "krwro_dco"   : KRWRO,
        "lf_i_bits"   : INT_BITS,
        "lf_f_bits"   : FRAC_BITS,
        "lf_params"   : lf_params,
        "init_params" : {
            "osc"   : main_pn_data["osc"].td[-1],
            "clk"   : main_pn_data["clk"].td[-1],
            "lf"    : main_pn_data["lf"].td[-1],
            "div"   : main_pn_data["div"].td[-1],
            "tdc"   : main_pn_data["tdc"].td[-1],
            "bbpd"  : main_pn_data["bbpd"].td[-1],
            "error" : main_pn_data["error"].td[-1],
            "kbbpd" : main_pn_data["kbbpd"].td[-1],
        },
    }

    spur_pn_data = pllsim_int_n(**spur_sim_params)

###############################################################################
# Plot data
###############################################################################

# plt.figure(3)

pn_sig = pn_signal(main_pn_data, DIV_N)
if RUN_SPURSIM: spur_pn_sig = pn_signal(spur_pn_data, DIV_N)
# pn_sig = gain(subtract(main_pn_data["div"], main_pn_data["clk"]), DIV_N)
# phase_noise = DIV_N*(main_pn_data["div"]-main_pn_data["clk"])
# spurs = DIV_N*(spur_pn_data["div"]-spur_pn_data["clk"])
# spur_pn_sig = gain(subtract(spur_pn_data["div"], spur_pn_data["clk"]), DIV_N)
# pn_sig_full = make_signal(td=phase_noise[int(len(phase_noise)/2):], fs=FS)
# spur_sig_full = make_signal(td=spurs, fs=FS_SPUR_SIM)
# pn_sig = make_signal(td=phase_noise[PLOT_SLICE], fs=FS)

# osc_freq = make_signal(td=FS*np.diff(main_pn_data["osc"][PLOT_SLICE])/(2*np.pi), fs=FS)
osc_freq = meas_inst_freq(main_pn_data["osc"])
# lf_sig = make_signal(td=main_pn_data["lf"][PLOT_SLICE], fs=FS)
# tdc_sig = make_signal(td=main_pn_data["tdc"][PLOT_SLICE], fs=FS)
# bbpd_sig = make_signal(td=main_pn_data["bbpd"][PLOT_SLICE], fs=FS)
# error_sig = make_signal(td=main_pn_data["error"][PLOT_SLICE], fs=FS)

# plt.subplot(2,3,1)
# plot_td(clk_sig, title="CLK")
# razavify()
plt.subplot(2,3,1)
plot_td(osc_freq, title="Inst. DCO Frequency", tmax=PLOT_SPAN)
# razavify()
plt.subplot(2,3,2)
plot_td(pn_sig, title="Phase error (noise)", tmax=PLOT_SPAN)
# razavify()
plt.subplot(2,3,3)
# plot_td(tdc_sig, title="TDC/PD Output", label="TDC", tmax=PLOT_SPAN)
# plot_td(bbpd_sig, title="BBPD Output", label="BBPD", tmax=PLOT_SPAN)
# plot_td(error_sig, title="Error Output", label="Combined Error", tmax=PLOT_SPAN)
plot_td(main_pn_data["tdc"], title="TDC/PD Output", label="TDC", tmax=PLOT_SPAN)
plot_td(main_pn_data["bbpd"], title="BBPD Output", label="BBPD", tmax=PLOT_SPAN)
plot_td(main_pn_data["error"], title="Error Output", label="Combined Error", tmax=PLOT_SPAN)
plt.legend()
# razavify()
plt.subplot(2,3,4)
# plot_td(lf_sig, title="Loop Filter Output", tmax=PLOT_SPAN)
plot_td(main_pn_data["lf"], title="Loop Filter Output", tmax=PLOT_SPAN)
# razavify()
plt.subplot(2,3,5)
# plot_pn_ssb2(spur_sig_full, dfmin=1e6, dfmax=PN_FMAX, line_fit=False, tmin=TSETTLE_EST)
# plot_pn_ssb2(pn_sig_full, dfmax=1e6, line_fit=False, tmin=TSETTLE_EST)
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=MAX_TSETTLE)
# plot_pn_ssb2(spur_pn_sig, dfmin=1e6, dfmax=PN_FMAX, line_fit=False)
plot_lf_ideal_pn(DCO_POWER, TEMP, fmax=PN_FMAX, **lf_params)
plt.grid()
if RUN_SPURSIM:
    plt.subplot(2,3,6)
    plt.plot(spur_pn_data["lf"].td)
plt.xlabel("n")
plt.ylabel("OTW[n]")
plt.title("Steady state oscillator tuning word")
plt.grid()
# plt.subplot(2,3,6)
# plt.plot(spur_pn_data["lf"])

# plot_fd(osc_sig_full)
# razavify()
adjust_subplot_space(2,2,0.1,0.1)
plt.show()

