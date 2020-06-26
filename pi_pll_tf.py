""" Plot PI-PLL response
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from libpll.plot import razavify
from libpll.filter import *

def _h(f, k, wz):
    s = 2j*np.pi*f
    return k*(1 + s/wz)/(s**2 + s*k/wz + k)
h = np.vectorize(_h, otypes=[complex])

tol = 50/12000
tsettle = 20e-6
f = np.geomspace(1e2, 1e7, 1001)

fig = plt.gcf()
fig.set_size_inches(12, 4)
fig.set_dpi(150)

plt.subplot(1,2,1)
for damping in np.geomspace(0.5/np.sqrt(2), 1, 4):
    k = np.log(tol)**2/(damping**2*tsettle**2)
    wz = np.sqrt(k)/(2*damping)
    print("K=%E, wz=%E"%(k,wz))
    g = h(f, k, wz)
    plt.semilogx(f, 20*np.log10(np.abs(g)), label="$\zeta$=%.3f"%damping)
    # plt.semilogx(f, 20*np.log10(np.abs((63/f)*(1-g))), label="$\zeta$=%.3f"%damping)
plt.grid()
plt.legend()
plt.title("PI-Controller PLL Closed-loop response")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude response [dB]")
razavify(loc="lower left", bbox_to_anchor=[0,0])
plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7], ["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$", ])
# plt.show()

t = np.arange(0, 40e-6, 1/16e6)
plt.subplot(1,2,2)

for damping in np.geomspace(0.5/np.sqrt(2), 1, 4):
    k = np.log(tol)**2/(damping**2*tsettle**2)
    wz = np.sqrt(k)/(2*damping)
    num = [k/wz, k]
    den = [1, k/wz, k]
    sys = scipy.signal.TransferFunction(num, den)
    t, x = scipy.signal.step(sys, T=t, N=len(t))
    plt.plot(t*1e6,x, label="$\zeta$=%.3f"%damping)
    print("K=%E, wz=%E"%(k,wz))
    tsettle = -np.log(tol)/min(abs(np.real(np.linalg.eigvals(sys._as_ss().A))))
    print("eigval tsettle=%E"%tsettle)
plt.grid()
plt.xticks([0, 10, 20, 30, 40])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
plt.legend()
plt.title("PI-Controller PLL Closed-loop step response")
plt.xlabel("Time [$\mu$s]")
plt.ylabel("Amplitude")
razavify(loc="lower right", bbox_to_anchor=[1,0])

plt.tight_layout()
plt.savefig("pi_response.pdf")
#plt.show()




tsettle = 20e-6
tol = 50/12000

damping = 1.0
k = np.log(tol)**2/(damping**2*tsettle**2)
wz = np.sqrt(k)/(2*damping)
print("K=%E, wz=%E"%(k,wz))



f = np.geomspace(1e2, 1e7, 1001)
print("bw", bw_pi_pll(k, wz/(2*np.pi)))
# foo()
tf_params = dict(k=k, fz=wz/(2*np.pi))
damps = np.linspace(0.1,1.0, 30)
powers = []
for damping in damps:
    k = np.log(tol)**2/(damping**2*tsettle**2)
    wz = np.sqrt(k)/(2*damping)
    tf_params = dict(k=k, fz=wz/(2*np.pi))
    powers.append(pll_pn_power_est(pi_pll_tf, tf_params, posc=5e-5, temp=293, m=150, n=150,
                       kdco=1e4, fclk=16e6, fmax=1e6, points=1000))
plt.plot(damps, powers)
plt.show()
print("**", opt_pll_tf_pi_controller( tsettle, tol, posc=1e-5, temp=293, m=150, n=150,
                                       kdco=1e4, fclk=16e6, fmax=1e6))
foo()

