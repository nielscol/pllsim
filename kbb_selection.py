import numpy as np
import matplotlib.pyplot as plt
from libpll.engine import run_sim, pllsim_int_n
from libpll.pncalc import ro_rw_model_param
from libpll.filter import opt_pll_tf_so_type2, lf_from_pll_tf, opt_pll_tf_pi_controller
from libpll.filter import opt_lf_num_bits, pll_pn_power_est, pll_tf, pi_pll_tf
from libpll.tools import timer, fixed_point, binary_lf_coefs
from libpll.analysis import pn_signal, meas_inst_freq, est_tsettle_pll_tf, noise_power_ar_model
from libpll.pncalc import min_ro_pn
from libpll.plot import plot_td, plot_pn_ar_model, plot_pn_ssb2, plot_lf_ideal_pn
from libpll.optimize import gss

FOSC = 2.4e9                # Oscillator frequency
FS = FCLK = 16e6             # Clock reference frequency
DIV_N = 150              # divider modulus
TDC_STEPS = 150           # 
USE_BBPD = True
BBPD_TSU = BBPD_TH = 10e-12

KDCO = 10e3             # DCO gain per OTW LSB
FL_DCO = 2.3e9          # starting frequency of DCO with fctrl=0
# LF_INIT = 16600         # Initial loop filter state (in=out=LF_INIT)
INIT_F_ERR = 12e6
SETTLE_DF = 10e3
MAX_TSETTLE = 50e-6
# K_OL = 1e10             # Open loop transfer function gain coefficient
LOOP_BW = 1e5           # PLL closed loop bandwidth
LOOP_DAMPING = 0.5      # PLL closed loop damping
#USE_SAVED_LF = False     #
# Phase noise 
# PN = -84.7                # dBc/Hz, target phase noise at PN_DF
DCO_POWER = 50e-6        # Calcuate PN based on RO physical limits
TEMP = 293                 # Simulation temperature, K
DCO_PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier

OPT_FMAX = 8e6

PLOT_SPAN = 50e-6
PN_FMAX = 1e8           # Limit max frequency of PN plot

SIM_STEPS = 100000


LF_MIN_BITS = 8
LF_MAX_BITS = 16
LF_RMS_FILT_ERROR = 0.1
LF_NOISE_FIGURE = 1 #dB - maximum increase in noise due to LF

#############################################################################3
# setup
DCO_PN = min_ro_pn(FOSC, DCO_PN_DF, DCO_POWER, TEMP)
print("\nOscillator characteristics:")
print("\tPower = %E, TEMP = %f"%(DCO_POWER, TEMP))
print("\tL(df=%f) = %.2f dBc"%(DCO_PN_DF, 10*np.log10(DCO_PN)))

print("\n*** Loop filter optimization under TDC-based feedback")
pll_tf_params = opt_pll_tf_pi_controller(MAX_TSETTLE, abs(SETTLE_DF/INIT_F_ERR), DCO_PN, DCO_PN_DF, TDC_STEPS,
                                  DIV_N, KDCO, FCLK, OPT_FMAX)
lf_params = lf_from_pll_tf(pll_tf_params, TDC_STEPS, DIV_N, KDCO, FCLK)
lf_params_bbpd = lf_params

_int_bits,_frac_bits = opt_lf_num_bits(lf_params, LF_MIN_BITS, LF_MAX_BITS, rms_filt_error=LF_RMS_FILT_ERROR, noise_figure=LF_NOISE_FIGURE)
INT_BITS = 32
FRAC_BITS = _frac_bits

binary_lf_coefs(lf_params, _int_bits, _frac_bits)

TSETTLE_EST = est_tsettle_pll_tf(pll_tf_params, SETTLE_DF/INIT_F_ERR)
LF_IDEAL = int(round((FOSC-FL_DCO)/KDCO))

KRWRO = ro_rw_model_param(f0=FOSC, power=DCO_POWER, temp=TEMP, n=SIM_STEPS, tstep=1.0/FS)

#############################################################################3
# sim

est_pow_pn = pll_pn_power_est(pi_pll_tf, lf_params, DCO_PN, DCO_PN_DF, TDC_STEPS, DIV_N, KDCO, FCLK, OPT_FMAX, plot=False)
print("est_pow_pn = %f"%est_pow_pn)
error = []
# for KBBPD in np.geomspace(0.01, 0.1, 11):
    # KBBPD = 0.03125
def kbbpd_cost(target_pow):
    def cost(KBBPD):
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
            "lf_params_bbpd"   : lf_params,
            "tsettle_est" : TSETTLE_EST,
            "verbose"     : False,
            "init_params" : {
                "osc"   : 0,
                "clk"   : 0,
                "lf"    : LF_IDEAL,
                "div"   : 0,
                "tdc"   : 0,
                "bbpd"  : 1,
                "error" : 0,
                "kbbpd" : KBBPD,
            },
        }

        main_pn_data = pllsim_int_n(verbose=False, **main_sim_params)
        pn_sig = pn_signal(main_pn_data, DIV_N)

        # pow_pn = np.mean(pn_sig.td**2)
        # print(pow_pn)
        pow_pn = 20*np.mean((pn_sig.td)**2)
        print(KBBPD, pow_pn)
        # error.append(np.abs(pow_pn - est_pow_pn))
        return pow_pn
    return cost

cost = kbbpd_cost(est_pow_pn)

print(gss(cost, arg="KBBPD", params={}, _min=0, _max=0.3, max_iter=15))

for x,y in zip(np.geomspace(0.01, 0.1, 11), error):
    print(x, y)



plt.subplot(2,1,1)
plot_td(pn_sig)
plt.subplot(2,1,2)
plt.plot(main_pn_data["osc"].td -DIV_N*main_pn_data["clk"].td)
plt.show()
plot_pn_ssb2(pn_sig, dfmax=8e8, line_fit=False)
plot_pn_ar_model(pn_sig, p=200, tmin=0)
plot_lf_ideal_pn(DCO_PN, DCO_PN_DF, **lf_params)
print(noise_power_ar_model(pn_sig, fmax=FCLK/2, p=100))
plt.show()
