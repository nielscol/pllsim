""" DCO Quantization noise simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from libpll._signal import make_signal
from libpll.plot import plot_fd, plot_td, plot_pn_ssb
from libpll.pllcomp import *
from libpll.analysis import find_rw_k, meas_ref_spur, eval_model_pn, average_pn
import time

F0 = 1 # Hz
CYCLES = 10000
SA_CYC = 10
TSTEP = 1/float(F0*SA_CYC)

SA = CYCLES*SA_CYC

LSB_DF = 1e5/2.4e9 # Hz
FREF = 32e6/2.4e9 # Hz
PN_DF = 1e6/2.4e9
t = np.arange(SA)/float(SA_CYC)

plt.subplot(1,2,1)
data = []
for FREF in np.geomspace(1e-1, 1e-3, 5):
    data.append([])
    for LSB_DF in np.geomspace(1e-6, 1e-3, 7):
        TREF = 1/float(FREF)
        dith = np.zeros(SA)
        for n in range(SA):
            trel = n*TSTEP/TREF - int(n*TSTEP/TREF)
            if trel >= 0 and trel < 0.5:
                dith[n] = 1

        phase = np.zeros(SA)
        for n in range(1, SA):
            phase[n] = phase[n-1] + 2*np.pi*((F0-LSB_DF/2)+LSB_DF*dith[n])*TSTEP

        td = np.sin(phase)

        sig = make_signal(td=td, fs = F0*SA_CYC)

        # plot_pn_ssb(sig, f0=F0, fref=FREF)
        #print(20*np.log10(eval_model_pn(sig, F0, PN_DF)))
        data[-1].append(20*np.log10(meas_ref_spur(sig, F0, FREF)))
print(data)
data[-1][-1] = 0
# plt.xscale("log")
# plt.yscale("log")
plt.imshow(data, interpolation="gaussian", extent=[-6, -3, -3, -1])
plt.colorbar()
CS = plt.contour(data, levels=[-120, -100, -80, -60, -40, -20], extent=[-6,-3,-1,-3], colors="w")
plt.clabel(CS, inline=1, fontsize=8)
plt.xticks([-6, -5, -4, -3], ["1e-6", "1e-5", "1e-4", "1e-3"])
plt.yticks([-3, -2, -1], ["1e-3", "1e-2", "1e-1"])
plt.xlabel("DCO LSB Resolution f_lsb/f0")
plt.ylabel("Clock Frequency fclk/f0")
plt.title("Reference Spur Level (dBc)")


data = []
FREF = 16e6/2.4e9
for LSB_DF in np.geomspace(1e-6, 1e-3, 7):
    TREF = 1/float(FREF)
    dith = np.zeros(SA)
    for n in range(SA):
        trel = n*TSTEP/TREF - int(n*TSTEP/TREF)
        if trel >= 0 and trel < 0.5:
            dith[n] = 1

    phase = np.zeros(SA)
    for n in range(1, SA):
        phase[n] = phase[n-1] + 2*np.pi*((F0-LSB_DF/2)+LSB_DF*dith[n])*TSTEP

    td = np.sin(phase)

    sig = make_signal(td=td, fs = F0*SA_CYC)

    # plot_pn_ssb(sig, f0=F0, fref=FREF)
    #print(20*np.log10(eval_model_pn(sig, F0, PN_DF)))
    data.append(20*np.log10(meas_ref_spur(sig, F0, FREF)))

plt.subplot(1,2,2)
plt.semilogx(np.geomspace(1e-6, 1e-3, 7)*2.4e6, data)
plt.xlabel("DCO LSB Resolution [kHz]")
plt.ylabel("Reference Spur Level (dBc)")
plt.title("Reference Spurs, f0=2.4 GHz, fclk=16 MHz")
plt.grid()
plt.show()

plt.show()
####################################

F0 = 2.4e9 # Hz
CYCLES = 100000
SA_CYC = 10
TSTEP = 1/float(F0*SA_CYC)

SA = CYCLES*SA_CYC

LSB_DF = 1e4 # Hz
FREF = 16e6 # Hz
PN_DF = 1e6
t = np.arange(SA)/float(SA_CYC)

TREF = 1/float(FREF)

TRANS_DENSITY = 0.0001
data = []
for TRANS_DENSITY in np.geomspace(1e-6, 1e-2, 9):
    data.append([])
    for LSB_DF in np.geomspace(1e3, 1e6, 7):
        dith = np.zeros(SA)
        for n in range(1, SA):
            trel = n*TSTEP/TREF - int(n*TSTEP/TREF)
            if trel < 1/float(SA_CYC):
                dith[n] = int(dith[n-1])^np.random.binomial(1, TRANS_DENSITY)
            else:
                dith[n] = dith[n-1]

        phase = np.zeros(SA)
        for n in range(1, SA):
            phase[n] = phase[n-1] + 2*np.pi*((F0-LSB_DF/2)+LSB_DF*dith[n])*TSTEP

        td = np.sin(phase)

        sig = make_signal(td=td, fs = F0*SA_CYC)
        # plot_pn_ssb(sig, f0=F0, line_fit=True)
        # plt.show()
        data[-1].append(average_pn(sig, F0, PN_DF, bins=5))
# print(20*np.log10(eval_model_pn(sig, F0, PN_DF)))
#print(20*np.log10(meas_ref_spur(sig, F0, FREF)))
print(data)
# data[2][0] = 0
plt.imshow(data, interpolation="gaussian", extent=[3, 6, -2, -6])
plt.colorbar()
CS = plt.contour(data, levels=[-120, -100, -80, -60, -40, -20], extent=[3,6,-6,-2], colors="w")
plt.clabel(CS, inline=1, fontsize=8)
plt.xticks([3, 4, 5, 6], ["1e0", "1e1", "1e2", "1e3"])
plt.yticks([-6, -5, -4, -3, -2], ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"])
plt.xlabel("DCO LSB Resolution [kHz]")
plt.ylabel("LSB Transition Density")
plt.title("DCO Phase Noise (dBc/Hz), f0=2.4 GHz, fclk= 16 MHz\n PN Offset=1MHz, Stochastic toggling of DCO LSB")
plt.show()
