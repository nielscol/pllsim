import matplotlib.pyplot as plt
import numpy as np
from libpll.filter import pll_otf, bw_pi_pll
from libpll.optimize import gss
from libpll.plot import razavify

def phase_margin(tf_params, fmax):
    """ In degrees
    """
    def cost(f):
        return abs(pll_otf(f, **tf_params))
    fug = gss(cost, arg="f", params={}, _min=0, _max=fmax, target=1)
    print(fug)
    return 180+np.angle(pll_otf(fug, **tf_params), deg=True)


def opt_pll_tf_pi_ph_margin(damping, ph_margin, tsettle_tol, fclk):
    """ Optimize tsettle of PI-controller PLL damping for fixed phase margin and damping
        returns : tsettle of PLL with specified damping and phase margin
    """
    k = np.log(tsettle_tol)**2/(damping**2)
    fz = np.sqrt(k)/(2*damping*2*np.pi)
    tf_params = dict(_type=2, k=k, fz=fz, fp=np.inf, delay=0)
    return phase_margin(tf_params, fclk)

def opt_pll_tf_pi_ph_margin2(damping, ph_margin, tsettle_tol, fclk, tsettle):
    """ Optimize tsettle of PI-controller PLL damping for fixed phase margin and damping
        returns : tsettle of PLL with specified damping and phase margin
    """
    k = np.log(tsettle_tol)**2/(tsettle**2*damping**2)
    fz = np.sqrt(k)/(2*damping*2*np.pi)
    tf_params = dict(_type=2, k=k, fz=fz, fp=np.inf, delay=0)
    print(k, fz)
    return phase_margin(tf_params, fclk)
_x = np.vectorize(opt_pll_tf_pi_ph_margin2, otypes=[float])

damping = np.linspace(0.5,1,11)
for tsettle in np.linspace(10e-6, 100e-5, 11):
    pm = _x(damping, 0, 0.001, 16e6, tsettle)
    plt.plot(damping, pm)

plt.show()


#@timer
def opt_pll_tf_pi_controller_fast_settling(ph_margin, max_tsettle, tsettle_tol, fclk, oversamp=20):
    """ Optimized PI-controller PLL for phase noise and settling time.
        Subject to maximum settling time constrained by tsettle, tol.
        points=1025 for Romberg integration (2**k+1)
    """
    def cost(damping):
        return opt_pll_tf_pi_ph_margin(damping, ph_margin, tsettle_tol, fclk)
    opt_damping = gss(cost, arg="damping", params={},
                      _min=0, _max=1.0, target=ph_margin, conv_tol=1e-5)
    def cost(tsettle):
        k = np.log(tsettle_tol)**2/(opt_damping**2*tsettle**2)
        fz = np.sqrt(k)/(2*opt_damping*2*np.pi)
        return bw_pi_pll(k, fz)
    opt_tsettle = gss(cost, arg="tsettle", params={},
                      _min=0, _max=max_tsettle, target=fclk/oversamp)
    opt_bw = cost(opt_tsettle)
    print(opt_damping, opt_tsettle)
    #if opt_tsettle > max_tsettle:
    #    raise Exception("Error: It is not possible to achieve the specified phase margin and lock time. \
    #                    Specified tsettle=%E, actual=%E. Decrease phase margin and try again."%(max_tsettle, opt_tsettle))

    print("For fast settling: opt pi tsettle = %E, damping = %f, bw = %E"%(opt_tsettle, opt_damping,opt_bw))

    return pll_tf_pi_controller(opt_tsettle, tol, opt_damping)

ph_margin = 60
max_tsettle = 50e-6
tol = 1e5/2.4e9
fclk = 16e6

opt_pll_tf_pi_controller_fast_settling(ph_margin, max_tsettle, tol, fclk)


def cost(damping):
    x= opt_pll_tf_pi_ph_margin(damping, ph_margin, tol, fclk)
    # print(x)
    return x
_cost = np.vectorize(cost, otypes=[float])
damps = np.linspace(0, 1, 100)
plt.plot(damps, _cost(damps))
plt.xlim((0,1))
plt.ylim((0,80))
plt.xlabel("Damping ratio $\zeta$")
plt.ylabel("Phase margin [degrees]")
plt.title("PI-controller PLL phase margin versus damping $\zeta$")
plt.grid()
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
razavify()
plt.tight_layout()
plt.savefig("damping_vs_pm.pdf")
plt.show()
foo()

