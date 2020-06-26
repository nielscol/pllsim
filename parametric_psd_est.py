import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.signal
import scipy.integrate
import json

def autocorrel(x, maxlags=10):
    rxx = np.zeros(maxlags+1)
    for n in range(maxlags+1):
        if n == 0:
            rxx[n] = np.sum(x*x)
        else:
            rxx[n] = np.sum(x[n:]*x[:-n])
    return rxx

def ar_model(data, p):
    data_fd = np.fft.fft(data)/float(len(data))
    half_len = int(0.5*len(data))
    power_data = np.sum(np.abs(data_fd[:half_len])**2)/float(half_len)
    gamma = autocorrel(data, maxlags=p)
    # gamma = np.correlate(data, data, mode="same")
    # gamma = np.fft.fftshift(gamma)
    # print(rxx, gamma[:p+1])
    a = solve_yule_walker(gamma, p)
    w, h = scipy.signal.freqz([1], a, 1025) # 513 for romberg integration
    power_ar = scipy.integrate.romb(np.abs(h)**2, 1/len(h))
    b = [np.sqrt(power_data/power_ar)]
    return b, a

def solve_yule_walker(gamma, p):
    gamma_1_p = gamma[1:p+1]
    toeplitz_gamma = scipy.linalg.toeplitz(gamma[0:p])
    a_1_p = -np.dot(np.linalg.inv(toeplitz_gamma), gamma_1_p.T)
    a = np.ones(p+1)
    a[1:] = a_1_p
    return a


with open("./ex_phase_error.json", "r") as file:
    _data = np.array(json.loads(file.read()))

l = 100000
data = _data[-l:]
print(len(data))

S = np.fft.fft(data)/float(l)
half_len = int(0.5*len(S))
fbin = np.pi/half_len
w = np.arange(1, half_len)*fbin
plt.semilogx(w, 20*np.log10(np.abs(S[1:half_len])))

l = 100000
b, a = ar_model(data, p=100)
w, h = scipy.signal.freqz(b, a, np.geomspace(np.pi*1e-6, np.pi, 1001))
plt.semilogx(w, 20*np.log10(np.abs(h)), label="B")




plt.show()

plt.hist(data, bins=500)
print("stdev=%f"%np.std(data))
plt.show()
