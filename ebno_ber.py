""" Simulate BPSK and GMSK ber
"""

import numpy as np
import matplotlib.pyplot as plt
from libradio.modulation import v_gmsk_pulse

N = 5000000
eb_no_min = 0
eb_no_max = 10
steps = 11
eb_no_dbs = np.linspace(eb_no_min, eb_no_max, steps)
ber = np.full(steps, np.nan)


ts = 1.0
#BT = [0.1, 0.3, 0.5, 1.0]
bt = 0.5
time = np.linspace(-5, 5, 11)
# time = np.linspace(-2, 2, 5)

pulse = v_gmsk_pulse(time, bt, ts)
pulse /= sum(pulse)

for n, eb_no_db in enumerate(eb_no_dbs):
    eb_no=10.0**(eb_no_db/10.0)
    noise_std = 1/np.sqrt(2*eb_no)
    x = np.random.choice((-1,1), N)
    y = x + noise_std*np.random.randn(N)
    y_d = 2*(y >= 0) - 1
    errors = np.sum(x != y_d)
    ber[n] = 1.0 * errors/N

    print("Eb/No = %.1f dB -> BER = %E"%(eb_no_db, ber[n]))

plt.plot(eb_no_dbs, ber, 'bo', eb_no_dbs, ber, 'k', label="BPSK")

""" Not rigorously accurate, GMSK modulates phase,
"""

mod_index=0.5

for n, eb_no_db in enumerate(eb_no_dbs):
    eb_no=10.0**(eb_no_db/10.0)
    noise_std = 1/np.sqrt(2*eb_no)
    x = np.random.choice((-1,1), N)
    y = np.convolve(x,pulse,mode="same") + np.diff(noise_std*np.random.randn(N), prepend=[0])/(mod_index*np.pi)
    y_d = 2*(y >= 0) - 1
    errors = np.sum(x != y_d)
    ber[n] = 1.0 * errors/N

    print("Eb/No = %.1f dB -> BER = %E"%(eb_no_db, ber[n]))

plt.plot(eb_no_dbs, ber, 'bo', eb_no_dbs, ber, 'r', label="GMSK")
plt.axis([0, 10, 1e-6, 1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True)
plt.title('BPSK Modulation')
plt.show()
