import numpy as np
import matplotlib.pyplot as plt
from libpll.filter import *
from libpll.plot import plot_loop_filter, plot_dco_pn, plot_tdc_pn, plot_total_pn, plot_fd
from libpll.plot import plot_pn_ssb2, plot_pll_tf_step
from libpll.pncalc import min_ro_pn, make_pn_sig
from libpll.analysis import meas_inband_power, snr, est_tsettle_pll_tf
from libradio.modpsd import gmsk_psd
from libradio.transforms import mix

""" Plot phase noise for filter optimizer to match second order filter in closed loop PLL

    Note: Settling time is very non-optimal!
"""

pn_dco_df = 1e6
pn_dco = min_ro_pn(f0=2.4e9, df=pn_dco_df, power=50e-6, temp=293)


fn = 1e5
damping = 0.707
k = 3e10
d = 0.707
tf = opt_pll_tf_so_type2(fn, d)
tf = lf_from_pll_tf(tf, 64, 150, 1e4, 16e6)
DF_INITIAL = 5e6
DF_SETTLED = 5e4
print("Tsettle estimate=%E"%est_tsettle_pll_tf(tf, DF_SETTLED/DF_INITIAL))
plt.subplot(1,2,1)
plot_loop_filter(pn_dco=pn_dco, pn_dco_df=pn_dco_df, **tf)
plt.subplot(1,2,2)
plot_pll_tf_step(tf, tmax=2e-4)
plt.show()


""" Plot various damping coefficients of second order response filter DCO noise,
    also match PLL response with pole/zero tuning to so response, filter dco noise
"""

f=np.geomspace(1e1, 1e7, 1001)
print(bw_solpf(5e4, 0.5))
ro_pn = min_ro_pn(f0=2.4e9, df=f, power=50e-6, temp=293)
for d in [0.354,0.5, 0.707, 1.0]:
    g = solpf(f, fn, d)
    tf = opt_pll_tf_so_type2(fn, d)
    tf = lf_from_pll_tf(tf, 64, 150, 1e4, 16e6)
    # plt.subplot(1,3,1)
    plot_dco_pn(pn_dco=pn_dco, pn_dco_df=pn_dco_df, label="Fitted TF  d=%f"%d, **tf)
    plt.semilogx(f, 10*np.log10(np.abs(1-g)**2*ro_pn), label="SOLPF d=%f"%d)
    # plt.subplot(1,3,2)
    # plot_tdc_pn(posc=50e-6, temp=293, label="d=%f"%d, **tf)
    # plt.subplot(1,3,3)
    # plot_total_pn(posc=50e-6, temp=293, label="d=%f"%d, **tf)
    # plt.semilogx(f, 10*np.log10(ro_pn))
plt.legend()
plt.show()
foo()

ks = -np.geomspace(1e1, 1e15, 1001)
fps = calc_fp(fn, damping, ks)
fzs = calc_fz(fn, fps, ks)
print(fzs.shape)
plt.semilogx(-ks, fps)
plt.semilogx(-ks, fzs)
plt.show()
fp = calc_fp(fn, damping, k)
fz = calc_fz(fn, fp, k)
print(fp, fz)

f = np.geomspace(1,1e8, 101)
a = pll_ojtf(f, k=k, fz=fz, fp=fp, delay=0.0)
g = a/(1+a)

_g = lpf_so(f, 50e3, 0.707)

# plt.semilogx(f, 20*np.log10(np.abs(a)))
plt.semilogx(f, 20*np.log10(np.abs(_g)))
plt.semilogx(f, 20*np.log10(np.abs(g)))
plt.semilogx(f, 20*np.log10(np.abs(1-g)))
plt.show()
