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
from libpll.plot import plot_loop_filter, plot_lf_ideal_pn, plot_pn_ar_model
# from libpll.pllcomp import *
from libpll.engine import run_sim, pllsim_int_n
from libpll.pncalc import ro_rw_model_param
from libpll.filter import opt_pll_tf_so_type2, lf_from_pll_tf, opt_pll_tf_pi_controller
from libpll.filter import opt_lf_num_bits, opt_pll_tf_pi_controller_bbpd
from libpll.filter import opt_pll_tf_pi_controller_fast_settling
from libpll.tools import timer, fixed_point, binary_lf_coefs
from libpll.analysis import pn_signal, meas_inst_freq, est_tsettle_pll_tf, noise_power_ar_model
from libpll.pncalc import min_ro_pn
from libpll.bbpd import opt_kbbpd
import time
import json
from copy import copy
###############################################################################
# Simulation parameters
###############################################################################

FOSC = 2.4e9                # Oscillator frequency
FBIN = 1e1 #1e1              # Target frequency bin size (for PN spectrum)
FS = FCLK = 16e6             # Clock reference frequency
DIV_N = 150              # divider modulus
TDC_STEPS = 150           # 
USE_BBPD = True
BBPD_TSU = BBPD_TH = 10e-12# 2e-12

LF_MIN_BITS = 8
LF_MAX_BITS = 16
LF_RMS_FILT_ERROR = 0.1
LF_NOISE_FIGURE = 1 #dB - maximum increase in noise due to LF

KDCO = 10e3             # DCO gain per OTW LSB
FL_DCO = 2.3e9          # starting frequency of DCO with fctrl=0
INIT_F_ERR = -12e6
SETTLE_DF_TDC = 10e4
SETTLE_DF_BBPD = 10e3
TRANS_SAFETY_FACTOR = 1 # use extra time before switching gears
MAX_TSETTLE = 50e-6
INIT_LF_MODE_TDC = "phase_noise" # "speed" or "phase_noise"
PHASE_MARGIN = 60
# LOOP_BW = 1e5           # PLL closed loop bandwidth
# LOOP_DAMPING = 0.5      # PLL closed loop damping
# Phase noise 
DCO_POWER = 50e-6        # Calcuate PN based on RO physical limits
TEMP = 293                 # Simulation temperature, K
DCO_PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier


PLOT_SPAN = 50e-6
PN_FMAX = 1e8           # Limit max frequency of PN plot

N_REF_SPURS = 6
FBIN_SPUR_SIM = 1e3
RUN_SPURSIM = False

OPT_KBBPD = False #True
OPT_FMAX = 8e6 #FCLK/2 # 5e6 #
MAX_ITER_BBPD_OPT = 5

SAVE_DATA = True
SAVE_FILE = "lin_pll_simulation.pickle"

sigma_ph=0.1
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

# foo()

# if False: #USE_BBPD:
#     print("\n*** Loop filte optimization under BBPD-based feedback")
#     print("\tEquivalent BBPD steps/ref cycle = %.1f"%M_BBPD)
#     pll_tf_bbpd = opt_pll_tf_pi_controller(MAX_TSETTLE, abs(SETTLE_DF/INIT_F_ERR), DCO_PN, DCO_PN_DF, M_BBPD,
#                                       DIV_N, KDCO, FCLK, OPT_FMAX)
#     lf_params_bbpd = lf_from_pll_tf(pll_tf_bbpd, TDC_STEPS, DIV_N, KDCO, FCLK)
#     lf_params_bbpd = lf_params
# else: lf_params_bbpd = None
# lf_params_bbpd = lf_params


# plt.figure(1)
# plot_loop_filter(RO_POWER, TEMP, **lf_params)


###############################################################################
# Main phase noise simulation
###############################################################################

print("\n***********************************************")
print("* Running main phase noise simulation         *")
print("***********************************************")
SIM_STEPS = int(round(FS/FBIN)) # make simulation long enough to get desired fbin
print("\n* Simulation samples = %E Sa"%SIM_STEPS)
print("\t Fs = %E Hz"%FS)
print("\t Fbin = %E Hz"%FBIN)
KRWRO = ro_rw_model_param(f0=FOSC, power=DCO_POWER, temp=TEMP, n=SIM_STEPS, tstep=1.0/FS)

main_sim_params = {
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
    "tsettle_est" : TSETTLE_EST*TRANS_SAFETY_FACTOR,
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

main_pn_data = pllsim_int_n(**main_sim_params)


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
        "tsettle_est" : TSETTLE_EST,
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
if SAVE_DATA:
    import pickle
    to_save = {}
    to_save["lf_params"] = lf_params
    to_save["lf_params_bbpd"] = lf_params_bbpd
    to_save["sigma_ph"] = sigma_ph
    to_save["dco_pn"] = DCO_PN
    to_save["sim_data"] = main_pn_data
    pickle_out = open(SAVE_FILE,"wb")
    pickle.dump(to_save, pickle_out)
    pickle_out.close()

pn_sig = pn_signal(main_pn_data, DIV_N)
rpm = noise_power_ar_model(pn_sig, fmax=OPT_FMAX, p=200, tmin=MAX_TSETTLE)
print("\n* Residual phase modulation = %E"%rpm)
print("\ti.e. integrated phase noise power")

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
if USE_BBPD: plot_td(main_pn_data["bbpd"], title="BBPD Output", label="BBPD", tmax=PLOT_SPAN)
if USE_BBPD: plot_td(main_pn_data["error"], title="Error Output", label="Combined Error", tmax=PLOT_SPAN)
plt.legend()
# razavify()
plt.subplot(2,3,4)
# plot_td(lf_sig, title="Loop Filter Output", tmax=PLOT_SPAN)
plot_td(main_pn_data["lf"], title="Loop Filter Output", tmax=PLOT_SPAN)
plt.xlabel("n")
plt.ylabel("OTW[n]")
plt.title("Steady state oscillator tuning word")
# razavify()
plt.subplot(2,3,5)
# plot_pn_ssb2(spur_sig_full, dfmin=1e6, dfmax=PN_FMAX, line_fit=False, tmin=TSETTLE_EST)
# plot_pn_ssb2(pn_sig_full, dfmax=1e6, line_fit=False, tmin=TSETTLE_EST)
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=MAX_TSETTLE)
plot_pn_ar_model(pn_sig, p=200, tmin=MAX_TSETTLE)
# plot_pn_ssb2(spur_pn_sig, dfmin=1e6, dfmax=PN_FMAX, line_fit=False)
if USE_BBPD and OPT_KBBPD:
    print(lf_params_bbpd)
    plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, mode="bbpd", sigma_ph=sigma_ph, **lf_params_bbpd)
elif INIT_LF_MODE_TDC is "speed":
    plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, **lf_params_bbpd)
else: plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, **lf_params)
# plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, fmax=PN_FMAX, **lf_params_bbpd)
plt.grid()
if RUN_SPURSIM:
    plt.subplot(2,3,6)
    plt.plot(spur_pn_data["lf"].td)
plt.grid()
# plt.subplot(2,3,6)
# plt.plot(spur_pn_data["lf"])

# plot_fd(osc_sig_full)
# razavify()
adjust_subplot_space(2,2,0.1,0.1)
plt.show()

# with open("ex_phase_error.json", "w") as file:
    # file.write(json.dumps(list(pn_sig.td)))

#plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=MAX_TSETTLE)
#plot_pn_ar_model(pn_sig, p=100, tmin=MAX_TSETTLE)

# plt.legend()
# # plt.axhline(9990, color="r")
# # plt.axhline(10010, color="r")
# plot_td(main_pn_data["lf"], title="Loop Filter Output", tmax=50e-6)
# plt.title("PLL loop filter transient response")
# plt.ylabel("Loop filter output")
# plt.xlabel("Time [$\mu$s]")
# plt.ylim((8800, 10300))
# # plt.ylim((8800, 10300))
# # plt.plot((23e-6,23e-6),(8800, 9990), color="b", linestyle="--")
# # plt.text(25e-6, 10030, "Lock tolerance band", color="r", fontsize=12)
# # plt.text(23.5e-6, 9500, "Lock time", color="b", fontsize=12, rotation=90)
# razavify(loc="lower left", bbox_to_anchor=[0,0])
# plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])
# 
# plt.tight_layout()
# # plt.show()
# plt.savefig("trans_loop_filter.pdf")
# 
# 
# plt.clf()
# plt.cla()

# plt.legend()
# plot_td(osc_freq, title="Inst. DCO Frequency", tmax=PLOT_SPAN)
# plt.title("PLL output instantaneous frequency")
# plt.xlabel("Time [$\mu$s]")
# plt.ylabel("Frequency [GHz]")
# razavify(loc="lower left", bbox_to_anchor=[0,0])
# plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])
# plt.yticks([2.388e9, 2.39e9, 2.392e9, 2.394e9, 2.396e9, 2.398e9, 2.40e9, 2.402e9],
#            ["2.388", "2.390", "2.392", "2.394", "2.396", "2.398", "2.400", "2.402"])
# plt.tight_layout()
# # plt.show()
# plt.savefig("trans_inst_freq.pdf")

# plt.legend()
# plot_td(main_pn_data["tdc"], title="TDC/PD Output", label="TDC", tmax=PLOT_SPAN)
# if USE_BBPD: plot_td(main_pn_data["bbpd"], title="BBPD Output", label="BBPD", tmax=PLOT_SPAN)
# if USE_BBPD: plot_td(main_pn_data["error"], title="Error Output", label="Combined Error", tmax=PLOT_SPAN)
# plt.title("PLL TDC and BBPD output transient response")
# plt.xlabel("Time [$\mu$s]")
# plt.ylabel("Signal [LSB]")
# razavify(loc="upper right", bbox_to_anchor=[1,1])
# plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])
# 
# plt.tight_layout()
# # plt.show()
# plt.savefig("trans_tdc_bbpd.pdf")

plt.legend()
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=MAX_TSETTLE)
plot_pn_ar_model(pn_sig, p=200, tmin=MAX_TSETTLE)
# plot_pn_ssb2(spur_pn_sig, dfmin=1e6, dfmax=PN_FMAX, line_fit=False)
plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, **lf_params)
plt.title("PLL output phase noise PSD")
# plt.xlabel("Time [$\mu$s]")
# plt.ylabel("Signal [LSB]")
plt.ylabel("Relative Power [dBc/Hz]")
plt.xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$"])
razavify(loc="lower left", bbox_to_anchor=[0,0])
plt.xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$"])

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.savefig("trans_phase_noise.pdf")

