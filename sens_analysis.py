import numpy as np
import matplotlib.pyplot as plt

LN2 = np.log(2)

params_nom = dict(kn=100e-6, w=1e-6, l=1e-6, n=3, c=1e-15, gamma=0.07, vdd=0.8, vt0=0.3)
params_std = dict(kn=0.05, w=0.01, l=0.01, n=0.0, c=0.01, gamma=0.01, vdd=0.01, vt0=0.01)

def f_center(kn, w, l, n, c, gamma, vdd, vt0):
    return (kn/(4*LN2*n*c))*(w/l)*(vdd*(7/(8*LN2)-1+gamma/(2*LN2)-0.5*gamma)-vt0*(1/LN2-1))

def sensitivity(func, params_nom, params_std, step = 1e-3):
    f_nom = func(**params_nom)
    sq_sum = 0
    for k, v in params_nom.items():
        init_v = params_nom[k]
        params_nom[k] = init_v*(1+step)
        df_dk = (func(**params_nom)-f_nom)/(init_v*step)
        sq_sum += (df_dk*params_std[k]*init_v/f_nom)**2
        params_nom[k] = init_v
    return np.sqrt(sq_sum)

print("f_center = %.4f GHZ"%(f_center(**params_nom)*1e-9))
print("std = %E"%sensitivity(f_center, params_nom, params_std))
