import matplotlib.pyplot as plt
import numpy as np
from libpll.pllcomp import DCO
from libpll.pncalc import rw_gain
from libpll._signal import make_signal
from libpll.plot import plot_pn_ssb2, plot_pn_ar_model, razavify

f0 = 0
kdco = 1e4
fs = 16e6
dt = 1/fs

steps = 100000

pn = 10**(-80/10)
df = 1e6

krw = rw_gain(pn, df, steps, dt, m=1)
print("krw=", krw)
dco = DCO(kdco, f0, dt, krw=krw)

osc_sig = np.zeros(steps)
for n in range(steps):
    osc_sig[n] = dco.update(fctrl=0)

osc_sig = make_signal(td=osc_sig, fs=fs)

plot_pn_ssb2(osc_sig, line_fit=True)
#plot_pn_ar_model(osc_sig)
plt.legend()
plt.title("DCO SSB Phase Noise [dBc/Hz],\n $\mathtt{krw}$ fitted to $L(\Delta f=10^6)$ = -80 dBc/Hz")
razavify(loc="lower left", bbox_to_anchor=[0,0])
plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7], ["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$"])


plt.tight_layout()
plt.savefig("dco_rw_pn.pdf")
