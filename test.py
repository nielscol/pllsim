from libpll.filter import *
from libpll.plot import plot_loop_filter

pn_dco = 10**(-84/10)
pn_dco_df = 1e6
tsettle_max = 50e-6
fosc = 2.4e9
tsettle_ftol = 100e3
tsettle_tol = tsettle_ftol/fosc
m = 150
n = 150
kdco = 1e4
fclk = 16e6
fmax = 8e6



# attempt to optimize for BBPD??? The current approach doesn't seen convergent...
tf, sigma_ph = opt_pll_tf_pi_controller_bbpd(tsettle_max, tsettle_tol, pn_dco, pn_dco_df, n, kdco, fclk,
                                             fmax, points=1025, max_iter=15)

lf_params = lf_from_pll_tf(tf, 2*np.pi, n, kdco, fclk)
plot_loop_filter(pn_dco, pn_dco_df, mode="bbpd", sigma_ph=sigma_ph, **lf_params)
plt.show()

# from libpll.engine import make_sweep_sim_params, flatten, unflatten
# 
# p = {
#     "a" : 1,
#     "b" : 10,
#     "c" : 100,
# }
# 
# x = make_sweep_sim_params(p, ["a", "b","c"], [[1,2,3],[10,11,12],[100,101,102]])
# print(x)
# 
# f = flatten(x)
# print(len(f))
# print(f)
# u = unflatten(f, [3,3,3])
# print(u)
