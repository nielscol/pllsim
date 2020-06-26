from libpll.opt_pll import calc_lf_bbpd, calc_discrete_lf, design_filters, s0_osc
from libpll.opt_pll import k_fixed_bw, total_int_pn, alpha_opt
import numpy as np
import matplotlib.pyplot as plt
from libpll.plot import razavify
from libpll.optimize import gss
from libpll.opt_lf_bits import opt_lf_num_bits

BETA = 0.8947598824664404

FOM = -160
RMS_JIT = 1.4e-12
FREF = 16e6
FOSC = 816e6
ALPHA_MAX = 0.1
POWER = 80e-6
KDCO = 5e3

#////////////////////////////////////////////////////////////////////////////////////
# Test filter design.
#////////////////////////////////////////////////////////////////////////////////////
lfs = design_filters(FOM, RMS_JIT, FREF, FOSC, ALPHA_MAX, POWER, KDCO, beta=BETA)
bbpd_int_bits, bbpd_frac_bits = opt_lf_num_bits(lfs["bbpd"], min_bits=8, max_bits=16, noise_figure=0.1, plot=True)
sc_int_bits, sc_frac_bits = opt_lf_num_bits(lfs["sc"], min_bits=8, max_bits=16, noise_figure=0.1)
plt.show()
print("Test optimization, SC:")
print("\tN int bits  = %d"%sc_int_bits)
print("\tN frac bits = %d"%sc_frac_bits)
print("Test optimization, BBPD:")
print("\tN int bits  = %d"%bbpd_int_bits)
print("\tN frac bits = %d"%bbpd_frac_bits)
foo()
#////////////////////////////////////////////////////////////////////////////////////
# Plot Alpha versus normalized phase noise power
#////////////////////////////////////////////////////////////////////////////////////
s0 = s0_osc(fom=FOM, p=POWER, fosc=FOSC)
print("s0_osc=",s0)

alpha = np.linspace(0.0,0.25,10001)
k = k_fixed_bw(alpha=alpha, fref=16e6)
beta=0.9
int_pn = total_int_pn(s0_osc=s0, k=k, fref=16e6, beta=0.9)
int_pn[int_pn <= 0] = np.inf

print("argmin=", alpha[np.argmin(int_pn)])
print((0.5/np.pi)*np.sqrt(3+np.sqrt(10))/np.sqrt(6*np.pi))
print("min=", np.min(int_pn))
print(3*np.pi**2*np.sqrt(6*np.pi)*s0/(2*16e6))


def f(alpha):
    k = k_fixed_bw(alpha=alpha, fref=16e6)
    int_pn = total_int_pn(s0_osc=s0, k=k, fref=16e6, beta=BETA)
    return 1/int_pn
alpha_asymptote = gss(f, "alpha", {}, _min=0.1, _max=0.20)
print("asympt=", alpha_asymptote)

for beta in [BETA]:
    int_pn = total_int_pn(s0_osc=s0, k=k, fref=16e6, beta=beta)
    int_pn[int_pn <= 0] = np.inf
    print("%f\t->\t%f\t%f"%(beta, alpha_opt(beta), alpha[np.argmin(int_pn)]))
    plt.plot(alpha, int_pn/np.min(int_pn))
    plt.scatter(alpha[np.argmin(int_pn)], 1)
# plt.clf()
# plt.plot(alpha, int_pn)
# plt.show()
# foo()

# plt.legend()
plt.ylim((0,5))
plt.xlim((0,0.16))
plt.grid()
plt.axvline(alpha_asymptote, linestyle="--", color="red")
plt.text(alpha_asymptote*0.97, max(plt.ylim())*0.7, "Asymptote at $\\alpha$ = %.4f"%alpha_asymptote, rotation=90, color="red")
plt.text(alpha[np.argmin(int_pn)]*0.80, 1.2, "$\\alpha_{opt}$=%.5f"%alpha[np.argmin(int_pn)])
plt.xlabel("$\\alpha = BW_{loop}/f_{ref}$ [Hz/Hz]")
plt.ylabel("Normalized $\\sigma^2_{\Phi n}$ [rad$^2$]")
plt.title("Phase Noise Power versus $\\alpha$")
razavify()
ticks = np.linspace(0, 0.15, 6)
plt.xticks(ticks, ["%.2f"%x for x in ticks])
plt.show()

