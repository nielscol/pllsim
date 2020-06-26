import numpy as np
import matplotlib.pyplot as plt
from libpll.filter import pll_tf_pi_controller, pll_pn_power_est, pi_pll_tf, lf_from_pll_tf
from libpll.plot import *
damping = 0.707
tol = 0.0001
pn = 10**(-80/10)
pn_df=1e6
m=150
n=150
kdco=1e4
fmax=5e6
fclk=16e6
tsettle = 16e-6
div_jit = 5e-11

tf = pll_tf_pi_controller(tsettle, tol, damping)
lf = lf_from_pll_tf(tf, m, n, kdco, fclk)
# plot_lf_ideal_pn(pn, pn_df, **lf)
plot_lf_ideal_pn_full(div_jit, pn, pn_df, **lf)
plt.ylim((-140,-40))
plt.xlim((1e2, 1e7))
plt.title("PLL SSB Phase Noise Components")
razavify()
plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7], ["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$"])

plt.tight_layout()
plt.savefig("pn_comps.pdf")
