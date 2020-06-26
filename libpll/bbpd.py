import numpy as np
from libpll.pncalc import ro_rw_model_param
from libpll.filter import pll_pn_power_est, pi_pll_tf
from libpll.optimize import gss
from libpll.engine import pllsim_int_n
from libpll.analysis import pn_signal

""" This approach does not seem to work well with short simulation spans (<1e6)
"""

def kbbpd_cost(lf_params, dco_pn, dco_pn_df, dco_power, temp, tdc_steps, div_n, kdco, fclk,
               opt_fmax, bbpd_tsu, bbpd_th, int_bits, frac_bits, sim_steps=10000):
    krwro = ro_rw_model_param(f0=fclk*div_n, power=dco_power, temp=temp, n=sim_steps,
                              tstep=1.0/fclk)
    target_pow = pll_pn_power_est(pi_pll_tf, lf_params, dco_pn, dco_pn_df, tdc_steps,
                                  div_n, kdco, fclk, opt_fmax, plot=False)
    print("!!!!", target_pow)
    def cost(kbbpd):
        main_sim_params = {
            "fclk"        : fclk,
            "fs"          : fclk,
            "sim_steps"   : sim_steps,
            "div_n"       : div_n,
            "tdc_steps"   : tdc_steps,
            "use_bbpd"    : True,
            "kbbpd"       : kbbpd,
            "bbpd_tsu"    : bbpd_tsu,
            "bbpd_th"     : bbpd_th,
            "kdco"        : kdco,
            "fl_dco"      : fclk*div_n,
            "krwro_dco"   : krwro,
            "lf_i_bits"   : int_bits,
            "lf_f_bits"   : frac_bits,
            "lf_params"   : lf_params,
            "lf_params_bbpd"   : lf_params,
            "tsettle_est" : 0,
            "init_params" : {
                "osc"   : 0,
                "clk"   : 0,
                "lf"    : 0,
                "div"   : 0,
                "tdc"   : 0,
                "bbpd"  : 1,
                "error" : 0,
                "kbbpd" : kbbpd,
            },
        }

        main_pn_data = pllsim_int_n(verbose=False, **main_sim_params)
        pn_sig = pn_signal(main_pn_data, div_n)

        # pow_pn = np.mean(pn_sig.td**2)
        # print(pow_pn)
        pow_pn = 20*np.mean((pn_sig.td)**2)
        # print(kbbpd, pow_pn)
        # error.append(np.abs(pow_pn - est_pow_pn))
        return pow_pn
    return cost

def opt_kbbpd(lf_params, dco_pn, dco_pn_df, dco_power, temp, tdc_steps, div_n, kdco, fclk,
               opt_fmax, bbpd_tsu, bbpd_th, int_bits, frac_bits, sim_steps=10000,
               max_iter=15):
    cost = kbbpd_cost(lf_params, dco_pn, dco_pn_df, dco_power, temp, tdc_steps, div_n, kdco, fclk,
               opt_fmax, bbpd_tsu, bbpd_th, int_bits, frac_bits, sim_steps)

    return gss(cost, arg="kbbpd", params={}, _min=0, _max=0.3, max_iter=max_iter)
