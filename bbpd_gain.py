import matplotlib.pyplot as plt
import numpy as np
from libpll.optimize import gss

def _bbpd(ph, ph_dead):
    if abs(ph) < ph_dead/2:
        return np.random.choice((-1,1))
    elif ph >= ph_dead/2:
        return 1
    else:
        return -1
bbpd = np.vectorize(_bbpd, otypes=[float])

def _calc_kbbpd(sigma_ph, m, ph_dead, samples=10000):
    xph = np.random.normal(0, sigma_ph, samples)
    x = bbpd(ph, ph_dead)
    rxx = np.var(ph)
    rxy = np.mean(ph*x)
    k = rxy/rxx
    return M/(2*np.pi*k)
calc_kbbpd = np.vectorize(_calc_kbbpd, otypes=[float])


# phases = np.linspace(-np.pi, np.pi, 1001)
# plt.plot(phases, bbpd(phases, 0.314))
# plt.show()

sigma_ph = 0.05
t_dead = 500e-12
fclk = 16e6
ph_dead = 2*np.pi*t_dead*fclk
print("ph_dead = %f rad"% ph_dead)
print("sigma_ph = %f rad"%sigma_ph)

M = 150
M2PI = M/(2*np.pi)
N = 1000000
ph = np.random.normal(0, sigma_ph, N)
x = bbpd(ph, ph_dead)
M2PI = M/(2*np.pi)

var = np.var(ph)
cov = np.mean(ph*x)
print("\n*****************************************")
print("cov = %f, var = %f"%(cov,var))
k = cov/var
print("k = %f"%k)

print("pow ideal = %f"%np.var(k*ph))
print("pow bbpd = %f"%np.var(x))
print("diff pow = %f"%(np.var(x)-np.var(k*ph)))
print("Excess noise factor = %f"%(np.var(x)/np.var(k*ph)))
min_kbbpd = np.sqrt((1/12)/(np.var(x)-np.var(k*ph)))
print(min_kbbpd)
foo()
error = np.var(ph*M2PI - kbbpd*x)
print("error = %f"%error)


var = np.var(ph)
cov = np.mean(ph*x)
print("\n*****************************************")
print("cov = %f, var = %f"%(cov,var))
k = cov/var
print("k = %f"%k)
print(k/sigma_ph)
kbbpd = M/(2*np.pi*k)
print("kbbpd = %f"%kbbpd)

print("pow ideal = %f"%np.var(M2PI*ph))
print("pow bbpd = %f"%np.var(x*kbbpd))
print("diff pow = %f"%(np.var(x*kbbpd)-np.var(M2PI*ph)))
print("Excess noise factor = %f"%(np.var(x*kbbpd)/np.var(M2PI*ph)))

error = np.var(ph*M2PI - kbbpd*x)
print("error = %f"%error)

def cost(kbbpd):
    return np.mean((ph*M2PI - kbbpd*x)**2)
opt_kbbpd = gss(cost, "kbbpd", {}, _min=kbbpd*0.1, _max=kbbpd*10)
print("opt kbbpd = %f"%opt_kbbpd)

print("\n*****************************************")
print("MSE optimization")

print("pow ideal = %f"%np.var(M2PI*ph))
print("pow bbpd = %f"%np.var(x*opt_kbbpd))
print("diff pow = %f"%(np.var(x*opt_kbbpd)-np.var(M2PI*ph)))
print("Excess noise factor = %f"%(np.var(x*opt_kbbpd)/np.var(M2PI*ph)))

error = np.var(ph*M2PI - opt_kbbpd*x)
print("error = %f"%error)

# plt.plot(x*kbbpd, label="BBPD")
# plt.plot(M2PI*ph, label="ideal")
# plt.legend()
# plt.show()
