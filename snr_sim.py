import numpy as np
import matplotlib.pyplot as plt
from libpll.filter import *
from libpll.plot import plot_loop_filter, plot_ro_pn, plot_tdc_pn, plot_total_pn, plot_fd
from libpll.plot import plot_pn_ssb2
from libpll.pncalc import min_ro_pn, make_pn_sig
from libpll.analysis import meas_inband_power, snr
from libradio.modpsd import gmsk_psd
from libradio.transforms import mix

FBIN = 1e2
FMAX = 3e6
BITRATE = 1e6
BT = 0.3
FN = 5e4
DAMPING = 0.707
M = 150
N = 150
KDCO = 1e4
FCLK = 16e6
POSC = 50e-6
TEMP = 293

mod = gmsk_psd(FBIN, FMAX, BITRATE, BT)
tf = opt_pll_tf_so_type2(FN, DAMPING)
tf = lf_from_pll_tf(tf, M, N, KDCO, FCLK)
pn_sig = make_pn_sig(FBIN, FMAX, POSC, TEMP, **tf)

bb_pn = mix(mod, pn_sig)
plot_fd(mod)
plot_fd(pn_sig)
plot_fd(bb_pn)

print(np.sum(pn_sig.fd/len(pn_sig.fd)))
print(pn_sig.fbin*np.sum(pn_sig.fd**2/len(pn_sig.fd)**2))
print(mod.fbin*np.sum(mod.fd**2/len(mod.fd)**2))

plt.show()
plt.plot(pn_sig.fbin*np.cumsum(pn_sig.fd**2/len(pn_sig.fd)**2))
plt.show()
foo()



DAMPINGS = np.geomspace(0.5/np.sqrt(2), np.sqrt(2), 5)
# DAMPINGS = [0.707]
FNS = np.geomspace(1e4, 1e6, 13)
for DAMPING in DAMPINGS:
    snrs = []
    for FN in FNS:
        print("FN=%f"%FN)
        mod = gmsk_psd(FBIN, FMAX, BITRATE, BT)
        # plot_fd(mod)
        tf = opt_otf_so_type2(FN, DAMPING)
        tf = otf_to_lf(tf, M, N, KDCO, FCLK)
        pn_sig = make_pn_sig(FBIN, FMAX, POSC, TEMP, **tf)

        bb_pn = mix(mod, pn_sig)
        snrs.append(10*np.log10(snr(mod, bb_pn, 0, BITRATE)))
        print("\tSNR=%f"%snrs[-1])
        # print(meas_inband_power(mod, 0, BITRATE))
        # print(meas_inband_power(bb_pn, 0, BITRATE))

    plt.semilogx(FNS, snrs, label="$\zeta$=%.2f"%DAMPING)
plt.grid(True)
plt.legend()
plt.xlabel("$\omega_n$")
plt.ylabel("Simulated SNR [dB]")
plt.title("Simulated SNR in Baseband, BT=0.3 GMSK\nat 1 Mbps with PLL Phase noise")
plt.show()
