import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

f0 = 2.448e9

SAMPLES = 10000000
SAMPLES_CYCLE = 10
N = 153

DF = 1e4/f0

t1 = np.arange(SAMPLES)/float(SAMPLES_CYCLE)
t2 = np.arange(SAMPLES)/float(SAMPLES_CYCLE*N)

dt = 1/float(SAMPLES_CYCLE)

mod = np.sign(np.sin(2*np.pi*t2))
mod = np.zeros(SAMPLES)
ref_samp = N*SAMPLES_CYCLE
sign = 1
for n in range(SAMPLES):
    if n%ref_samp == 0:
        sign *= -1
    mod[n] = sign

ph = np.zeros(SAMPLES)
print("sign[0]", mod[0])
ph[0] = np.pi*N*DF
for n in range(1, SAMPLES):
    ph[n] = ph[n-1] + 2*np.pi*(1+DF*mod[n])*dt
plt.plot(ph-2*np.pi*t1)
plt.show()

sig1 = np.sin(ph)
sig2 = np.sin(2*np.pi*t1)

# plt.plot(mod)
# plt.plot(np.diff(ph))
# plt.plot(sig1)
# plt.plot(sig2)
# plt.show()
# bartlett = np.bartlett(SAMPLES)
bh = scipy.signal.blackmanharris(SAMPLES)
SIG1 = np.fft.fft(sig1*bh)
SIG2 = np.fft.fft(sig2*bh)

offset = int(SAMPLES/SAMPLES_CYCLE)+1
print (offset)
HALF_SPAN = int(SAMPLES/2)
f = np.arange(1, HALF_SPAN-offset+1 )*(SAMPLES_CYCLE/SAMPLES)*f0

nom_pow = np.abs(SIG1[offset-1])

# plt.plot(20*np.log10(np.abs(SIG2)/nom_pow))
# plt.plot(20*np.log10(np.abs(SIG1)/nom_pow))
# plt.show()
plt.semilogx(f, 20*np.log10(np.abs(SIG1[offset:HALF_SPAN])/nom_pow))
# plt.semilogx(f, 20*np.log10(np.abs(SIG2[offset:HALF_SPAN])/nom_pow))
plt.xlabel("$\Delta$f from carrier [Hz]")
plt.ylabel("dBc/Hz")
plt.show()
