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
# TRANS_SAFETY_FACTOR = 2 # use extra time before switching gears
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

OPT_KBBPD = False
OPT_FMAX = 5e6 #FCLK/2 # 5e6 #
MAX_ITER_BBPD_OPT = 5

SAVE_DATA = False
SAVE_FILE = "bbpd_pll_simulation.pickle"


import pickle
pickle_in = open(SAVE_FILE,"rb")
sim_data = pickle.load(pickle_in)
lf_params = sim_data["lf_params"]
lf_params_bbpd = sim_data["lf_params_bbpd"]
sigma_ph = sim_data["sigma_ph"]
DCO_PN = sim_data["dco_pn"]
main_pn_data = sim_data["sim_data"]

if False:
    plt.legend()
    # plt.axhline(9990, color="r")
    # plt.axhline(10010, color="r")
    plot_td(main_pn_data["lf"], title="Loop Filter Output", tmax=50e-6)
    plt.title("PLL loop filter transient response")
    plt.ylabel("Loop filter output")
    plt.xlabel("Time [$\mu$s]")
    plt.ylim((9000, 10500))
    # plt.ylim((8800, 10300))
    # plt.plot((23e-6,23e-6),(8800, 9990), color="b", linestyle="--")
    # plt.text(25e-6, 10030, "Lock tolerance band", color="r", fontsize=12)
    # plt.text(23.5e-6, 9500, "Lock time", color="b", fontsize=12, rotation=90)
    razavify(loc="lower left", bbox_to_anchor=[0,0])
    plt.xticks([0, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    plt.yticks([9000, 9200, 9400, 9600, 9800, 10000, 10200, 10400], ["9000", "9200", "9400", "9600", "9800", "10000", "12000", "14000"])
    plt.xlim((0, 2e-5))
    plt.tight_layout()
    plt.savefig("trans_loop_filter_fast.pdf")
    plt.show()
    # foo()
    foo()
# 
# 
# plt.clf()
# plt.cla()
if False:
    osc_freq = meas_inst_freq(main_pn_data["osc"])
    plt.legend()
    plot_td(osc_freq, title="Inst. DCO Frequency", tmax=PLOT_SPAN)
    plt.title("PLL output instantaneous frequency")
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Frequency [GHz]")
    razavify(loc="lower left", bbox_to_anchor=[0,0])
    # plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])
    plt.xlim((0, 2e-5))
    plt.xticks([0, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
    plt.yticks([2.388e9, 2.39e9, 2.392e9, 2.394e9, 2.396e9, 2.398e9, 2.40e9, 2.402e9, 2.404e9],
               ["2.388", "2.390", "2.392", "2.394", "2.396", "2.398", "2.400", "2.402", "2.404"])
    plt.tight_layout()
    plt.savefig("trans_inst_freq_fast.pdf")
    plt.show()
    foo()

if False:
    plt.legend()
    plot_td(main_pn_data["tdc"], title="TDC/PD Output", label="TDC", tmax=PLOT_SPAN)
    if USE_BBPD: plot_td(main_pn_data["bbpd"], title="BBPD Output", label="BBPD", tmax=PLOT_SPAN)
    if USE_BBPD: plot_td(main_pn_data["error"], title="Error Output", label="Combined Error", tmax=PLOT_SPAN)
    plt.title("PLL TDC and BBPD output transient response")
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Signal [LSB]")
    razavify(loc="upper right", bbox_to_anchor=[1,1])
    plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])
    plt.xlim((0, 2e-5))
    plt.xticks([0, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])

    plt.tight_layout()
    plt.savefig("trans_tdc_bbpd_fast.pdf")
    plt.show()

    foo()

pn_sig = pn_signal(main_pn_data, DIV_N)
plt.legend()
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=MAX_TSETTLE)
plot_pn_ar_model(pn_sig, p=200, tmin=MAX_TSETTLE)
# plot_pn_ssb2(spur_pn_sig, dfmin=1e6, dfmax=PN_FMAX, line_fit=False)
# plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, **lf_params)
plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, mode="bbpd", sigma_ph=sigma_ph, **lf_params_bbpd)
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
plt.savefig("trans_phase_noise_fast.pdf")
plt.show()
plt.xlim((0, 2e-5))
plt.xticks([0, 0.5e-5, 1e-5, 1.5e-5, 2e-5], ["0", "5", "10", "15", "20"])
