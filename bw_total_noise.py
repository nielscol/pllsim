import numpy as np
import matplotlib.pyplot as plt
from libpll.filter import pll_tf_pi_controller, pll_pn_power_est, pi_pll_tf
from libpll.plot import razavify

damping = 0.707
tol = 0.0001
pn = 10**(-80/10)
pn_df=1e6
m=150
n=150
kdco=1e4
fmax=5e6
fclk=16e6


lfs = []
for tsettle in np.geomspace(2e-6, 2e-4, 101):
    lf = pll_tf_pi_controller(tsettle, tol, damping)
    lfs.append(lf)

bws = []
int_pns = []
for lf in lfs:
    int_pn = pll_pn_power_est(pi_pll_tf, lf, pn, pn_df, m, n, kdco, fclk, fmax)
    int_pns.append(int_pn)
    bws.append(lf["bw"])
tsettles = np.geomspace(2e-6, 2e-4, 101)
bw_opt = bws[np.argmin(int_pns)]
int_pn_opt = int_pns[np.argmin(int_pns)]
tsettle_opt = tsettles[np.argmin(int_pns)]
print(tsettle_opt)
plt.semilogx(bws, int_pns)
plt.scatter([bw_opt],[int_pn_opt])
plt.text(150000, 0.28, "(%.1fe3, %.3f)"%(bw_opt/1000, int_pn_opt))
plt.grid()
plt.title("PLL total phase noise versus loop bandwidth")
plt.ylim((0.2,1.2))
plt.xlim((3e4, 3e6))
plt.yticks([.2,.4,.6,.8,1,1.2],["0.2","0.4","0.6","0.8","1.0","1.2"])
razavify()
ax = plt.gca()
print(ax.texts)
plt.xticks([1e5, 1e6], ["$10^5$", "$10^6$"])
plt.ylabel("Integrated phase noise power [rad$^2$]")
plt.xlabel("PLL $f_{-3dB}$ Bandwidth [Hz]")
plt.tight_layout()
plt.savefig("bandwidth_vs_pn.pdf")
# plt.show()
