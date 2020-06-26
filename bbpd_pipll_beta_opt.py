"""BETA OPT  SIM BBPD PI PLL
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from copy import copy
from libpll.engine import sim_bbpd_pipll
from libpll.tools import timer, binary_lf_coefs, debias_pn
from libpll._signal import make_signal
from libpll.pncalc import min_ro_pn, fom_to_pn, rw_gain_fom
from libpll.opt_pll import *
from libpll.opt_lf_bits import opt_lf_num_bits
from libpll.plot import plot_fd, plot_td, plot_pn_ssb2, razavify, adjust_subplot_space
from libpll.plot import plot_loop_filter, plot_lf_ideal_pn, plot_pn_ar_model, plot_pi_pll_bbpd_pn
from libpll.plot import plot_osc_pn_ideal, plot_pi_pll_osc_pn
from libpll.analysis import pn_signal2, meas_inst_freq, est_tsettle_pll_tf, noise_power_ar_model
from libpll.optimize import gss
import multiprocessing


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

FOSC = 816e6                # Oscillator frequency
FBIN = 1e1 #1e1 #1e1              # Target frequency bin size (for PN spectrum)
FS = FCLK = 16e6             # Clock reference frequency
USE_BBPD = True
BBPD_RMS_JIT = 0.0e-12

LF_MIN_BITS = 8
LF_MAX_BITS = 16
LF_RMS_FILT_ERROR = 0.1 # dB
LF_NOISE_FIGURE = 0.1 #dB - maximum increase in noise due to LF
ALPHA_MAX = 0.10     # max BW/FCLK
MAX_ALPHA = 0.1
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
INIT_F_ERR = 0.0
SETTLE_DF_SCPD = 1e6
SETTLE_DF_BBPD = 1e5
MAX_TSETTLE = 50e-6
FORCE_GEAR_SW_T = True
GEAR_SW_T = 0.0
GEAR_SW_SAFETY_FACTOR = 8 # use extra time before switching gears

PN_CALC_SAFETY_FACTOR = 4 # times the gearswithc time

DCO_POWER = 80e-6        # Calcuate PN based on RO physical limits
TEMP = 300                 # Simulation temperature, K
DCO_PN_DF = 1e6             # Hz, phase noise measurement frequency offset from carrier
DCO_FOM = -160
USE_THEOR_DCO_PN = False    # Use the theoretical value for ring oscillator PN

PLOT_SPAN = 50e-6
PN_FMAX = 1e8           # Limit max frequency of PN plot

N_REF_SPURS = 6
FBIN_SPUR_SIM = 1e3
RUN_SPURSIM = False

"""BETA OPT  SIM BBPD PI PLL
"""

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Determine initial LF state
KDCO1 = KVCO1*VDD/2**DCO_FINE_BITS             # Fine DCO gain
KDCO2 = KVCO2*VDD/2**DCO_MED_BITS

print("KDCO1 (fine) = %.2E Hz/LSB"%KDCO1)
print("KDCO2 (med)  = %.2E Hz/LSB"%KDCO2)

PLOT_SLICE = slice(0, int(PLOT_SPAN*FS), 1) # matplotlib is slow with too many points, reduce for time domain
MED_IDEAL = np.floor((FOSC-FL_DCO)/KDCO2)
FINE_IDEAL = np.round((FOSC-FL_DCO-MED_IDEAL*KDCO2)/KDCO1)
LF_IDEAL = MED_IDEAL*2**(DCO_FINE_BITS) + FINE_IDEAL

MED_INIT = np.floor((FOSC-FL_DCO+INIT_F_ERR)/KDCO2)
FINE_INIT = np.round((FOSC-FL_DCO+INIT_F_ERR-MED_INIT*KDCO2)/KDCO1)
LF_INIT = MED_INIT*2**(DCO_FINE_BITS) + FINE_INIT

print("* LF_MAX   = %d"%(2**(DCO_FINE_BITS + DCO_MED_BITS)-1))
print("* LF_IDEAL = %d"%LF_IDEAL)
print("* LF_INIT  = %d"%LF_INIT)




#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Determine DCO phase noise

if USE_THEOR_DCO_PN:
    DCO_PN = min_ro_pn(FOSC, DCO_PN_DF, DCO_POWER, TEMP)
    DCO_FOM = 10*np.log10(DCO_PN*FOSC**2*DCO_POWER*1e3/DCO_PN_DF**2)
else:
    DCO_PN = fom_to_pn(DCO_FOM, DCO_POWER, FOSC, DCO_PN_DF)
print("\nOscillator characteristics:")
print("\tFOM = %.1F dB"%DCO_FOM)
print("\tPower = %.1f uW"%(DCO_POWER*1e6))
print("\tTEMP = %.1f K"%TEMP)
print("\tL(df=%.2E) = %.2f dBc/Hz"%(DCO_PN_DF, 10*np.log10(DCO_PN)))

KB = 1.38064852e-23

KRW = rw_gain_fom(fom_db=DCO_FOM, fosc=FOSC, power=DCO_POWER, fs=FS)
print("\tkrw = %.2E"%KRW)

# initial design
lfs = design_filters(DCO_FOM, BBPD_RMS_JIT, FCLK, FOSC, ALPHA_MAX, DCO_POWER, KDCO1)


ITERS = 10
beta_opt = 0.8947598824664404
SIM_PN_RMS = np.zeros(ITERS)
EST_PN_RMS = np.zeros(ITERS)
BETAS = np.zeros(ITERS)
ALPHAS = np.zeros(ITERS)
for n in range(ITERS):
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Design loop filters



    lfs = design_filters(DCO_FOM, BBPD_RMS_JIT, FCLK, FOSC, ALPHA_MAX, DCO_POWER, KDCO1, beta=beta_opt)
    bbpd_int_bits, bbpd_frac_bits = opt_lf_num_bits(lfs["bbpd"], min_bits=LF_MIN_BITS, max_bits=LF_MAX_BITS, noise_figure=LF_NOISE_FIGURE)
    sc_int_bits, sc_frac_bits = opt_lf_num_bits(lfs["sc"], min_bits=LF_MIN_BITS, max_bits=LF_MAX_BITS, noise_figure=LF_NOISE_FIGURE)
    print("Test optimization, SC:")
    print("\tN int bits  = %d"%sc_int_bits)
    print("\tN frac bits = %d"%sc_frac_bits)
    print("Test optimization, BBPD:")
    print("\tN int bits  = %d"%bbpd_int_bits)
    print("\tN frac bits = %d"%bbpd_frac_bits)
    if not FORCE_INT_BITS:
        INT_BITS = max((sc_int_bits, bbpd_int_bits))
    if not FORCE_FRAC_BITS:
        FRAC_BITS = max((sc_frac_bits, bbpd_frac_bits))
    print("\nFinal Dataword size:")
    print("\tN int bits  = %d"%INT_BITS)
    print("\tN frac bits = %d"%FRAC_BITS)


    print("\n* Final resolution of data words: %d frac bits, %d int bits, 1 sign bit"%(FRAC_BITS, INT_BITS))
    print("\n* Digitized ilter coeffients in SCPD mode")
    binary_lf_coefs(lfs["sc"], INT_BITS, FRAC_BITS)
    print("\n* Digitized filter coeffients in BBPD mode")
    binary_lf_coefs(lfs["bbpd"], INT_BITS, FRAC_BITS)

    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Estimate lock times
    sc_tsettle = tsettle_lf(lfs["sc"], ftol=SETTLE_DF_SCPD, df0=INIT_F_ERR)
    bbpd_tsettle = tsettle_lf(lfs["bbpd"], ftol=SETTLE_DF_BBPD, df0=SETTLE_DF_SCPD)
    EST_TOTAL_LOCK_TIME = sc_tsettle + bbpd_tsettle
    print("\n* Final estimated settling time of PLL (SC and or BBPD mode) = %E s"%EST_TOTAL_LOCK_TIME)
    print("\n* Estimated settling time of PLL (SC) = %E s"%sc_tsettle)


    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Run simulation
    """BETA OPT SIM BBPD PI PLL
    """



    print("\n***********************************************")
    print("* Running main phase noise simulation         *")
    print("***********************************************")
    SIM_STEPS = int(round(FS/FBIN)) # make simulation long enough to get desired fbin
    print("\n* Simulation samples = %.2E Sa"%SIM_STEPS)
    print("\t Fs = %.2E Hz"%FS)
    print("\t Fbin = %.2E Hz"%FBIN)
    SIM_SPAN = SIM_STEPS/float(FS)

    if not FORCE_GEAR_SW_T:
        GEAR_SW_T = sc_tsettle*GEAR_SW_SAFETY_FACTOR

    main_sim_params = {
        "ignore_clk"      : True,
        "fclk"            : FCLK,
        "fs"              : FS,
        "sim_steps"       : SIM_STEPS,
        "div_n"           : int(round(FOSC/FCLK)),
        "use_bbpd"        : USE_BBPD,
        "bbpd_rms_jit"    : BBPD_RMS_JIT,
        "kdco1"           : KDCO1,
        "kdco2"           : KDCO2,
        "fine_bits"       : DCO_FINE_BITS,
        "med_bits"        : DCO_MED_BITS,
        "fl_dco"          : FL_DCO,
        "krw_dco"         : KRW,
        "lf_i_bits"       : INT_BITS,
        "lf_f_bits"       : FRAC_BITS,
        "lf_params_sc"    : lfs["sc"],
        "lf_params_bbpd"  : lfs["bbpd"],
        "tsettle_est"     : GEAR_SW_T,
        "init_params"     : {
            "osc"   : 0,
            "clk"   : 0,
            "lf"    : LF_INIT,
            "scpd"   : 0,
            "bbpd"  : 1,
            "error" : 0,
            "med"   : MED_INIT,
            "fine"  : FINE_INIT,
        },
    }

    """ABETA OPT SIM BBPD PI PLL
    """

    sim_data = sim_bbpd_pipll(**main_sim_params)


    pn_sig = pn_signal2(sim_data, div_n=round(FOSC/FCLK))
    pn_sig = debias_pn(pn_sig, 0.0)
    # plot_td(pn_sig, tmax=PLOT_SPAN, title="Phase error", dots=True)

    half = int(SIM_STEPS/2)
    rms = np.std(pn_sig.td)
    print("\nSimulated RMS PN:\t\t%.2E rad -> %.2f dB"%(rms, 20*np.log10(rms)))
    est_rms_pn_design = np.sqrt(lfs["bbpd"]["int_pn"])
    print("Filter design RMS PN Est:\t%.2E rad -> %.2f dB"%(est_rms_pn_design, 20*np.log10(est_rms_pn_design)))

    SIM_PN_RMS[n] = rms
    EST_PN_RMS[n] = est_rms_pn_design

    def f(beta, target_pn):
        lf = design_filters(DCO_FOM, 0, FCLK, FOSC, MAX_ALPHA, DCO_POWER, KDCO1, beta)
        return abs(target_pn-lf["bbpd"]["int_pn"])

    beta_opt = gss(f, "beta", {"target_pn":rms**2}, _min=0.5, _max=1.5) #1.0088433280171367 #
    alpha = alpha_opt(beta_opt)

    print("ITER %d RESULT:\tbeta = %.3E,\talpha = %.3E,\trms_pn = %.3E"%(n, beta_opt, alpha, rms))
    BETAS[n] = beta_opt
    ALPHAS[n] = alpha
print("\nITER\tBETA\tALPHA\trms")
for n, v in enumerate(BETAS):
    print("%d\t%.8f\t%.8f\t%E"%(n, v, ALPHAS[n], SIM_PN_RMS[n]))
# plt.plot(ALPHAS, SIM_PN_RMS**2)
# plt.plot(ALPHAS, EST_PN_RMS**2)
# plt.show()
foo()

plt.figure(1)
plt.subplot(3,3,1)
plot_td(sim_data["lf"], tmax=PLOT_SPAN, title="LF", dots=True)
plt.subplot(3,3,2)
plot_td(sim_data["scpd"], tmax=PLOT_SPAN, title="SCPD", dots=True)
plt.subplot(3,3,3)
plot_td(sim_data["bbpd"], tmax=PLOT_SPAN, title="BBPD", dots=True)
plt.subplot(3,3,4)
plot_td(meas_inst_freq(sim_data["osc"]), tmax=PLOT_SPAN, title="Freq", dots=True)
plt.subplot(3,3,5)
plot_td(sim_data["error"], tmax=PLOT_SPAN, title="Error", dots=True)
plt.subplot(3,3,6)
plot_td(sim_data["fine"], tmax=PLOT_SPAN, title="fine", dots=True)
plt.subplot(3,3,7)
plot_td(sim_data["med"], tmax=PLOT_SPAN, title="med", dots=True)
plt.subplot(3,3,8)
pn_sig = pn_signal2(sim_data, div_n=round(FOSC/FCLK))
pn_sig = debias_pn(pn_sig, 0.5*SIM_SPAN)
plot_td(pn_sig, tmax=PLOT_SPAN, title="Phase error", dots=True)

half = int(SIM_STEPS/2)
rms = np.std(pn_sig.td[half:])
print("\nSimulated RMS PN:\t\t%.2E rad -> %.2f dB"%(rms, 20*np.log10(rms)))
est_rms_pn_design = np.sqrt(lfs["bbpd"]["int_pn"])
print("Filter design RMS PN Est:\t%.2E rad -> %.2f dB"%(est_rms_pn_design, 20*np.log10(est_rms_pn_design)))

x = np.zeros(SIM_STEPS+1)
x[1:] = sim_data["lf"].td
em =  -np.cumsum(2*np.pi*KDCO1*np.diff(x)*(1/float(FS)))
em_pn = make_signal(em, fs=FS)
em_pn = debias_pn(em_pn, 0.5*SIM_SPAN)
plot_td(em_pn, tmax=PLOT_SPAN, title="Emergent PN", dots=True)

plt.subplot(3,3,9)
pn = copy(pn_sig.td[int(np.ceil(GEAR_SW_T*FCLK)):])
em = copy(em_pn.td[int(np.ceil(GEAR_SW_T*FCLK)):])
pn[np.abs(pn)>3*rms] = np.nan
em[np.abs(em)>3*rms] = np.nan
plt.hist(pn, alpha=0.5, bins=50, density=True)
plt.hist(em, alpha=0.5, bins=50, density=True)


plt.figure(2)
plot_pn_ssb2(pn_sig, dfmax=PN_FMAX, line_fit=False, tmin=GEAR_SW_T*PN_CALC_SAFETY_FACTOR)
plot_pn_ar_model(pn_sig, p=200, tmin=GEAR_SW_T*PN_CALC_SAFETY_FACTOR)
# rpm = noise_power_ar_model(pn_sig, fmax=FCLK/2, p=200, tmin=GEAR_SW_T)
# print("int rpm = ", np.sqrt(rpm))


plot_pn_ar_model(em_pn, p=200, tmin=GEAR_SW_T*PN_CALC_SAFETY_FACTOR)
plot_pi_pll_bbpd_pn(**lfs["bbpd"])
plot_pi_pll_osc_pn(DCO_PN, DCO_PN_DF, **lfs["bbpd"])
# plot_osc_pn_ideal(DCO_PN, DCO_PN_DF)
plt.show()

# plt.scatter(em_pn.td[int(np.ceil(GEAR_SW_T*FCLK)):],pn_sig.td[int(np.ceil(GEAR_SW_T*FCLK)):])
# plt.show()
