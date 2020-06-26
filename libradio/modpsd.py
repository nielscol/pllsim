""" Compute the power spectral density of several modulation schemes.
"""
import numpy as np
import matplotlib.pyplot as plt
from libradio._signal import make_signal

def _calc_msk(f):
    if abs(f) == 0.25: return 1
    else: return (16/(np.pi**2))*(np.cos(2*np.pi*f))**2/((1 - 16*f**2)**2)
calc_msk = np.vectorize(_calc_msk, otypes=[float])

def msk_psd(fbin, fmax, bitrate):
    bins = int(round(2*fmax/fbin))
    f = fbin*(np.arange(bins)-int(bins/2))
    f_norm = f/bitrate
    psd = calc_msk(f_norm)
    psd /= np.sum(psd)
    return make_signal(fd=np.sqrt(np.fft.fftshift(psd)), fs=bins*fbin)


def gmsk_psd(fbin, fmax, bitrate, bt):
    bins = int(round(2*fmax/fbin))
    f = fbin*(np.arange(bins)-int(bins/2))
    f_norm = f/bitrate
    gauss = np.exp((np.log(2)/-2)*(f_norm/bt)**2)
    psd = gauss*calc_msk(f_norm)
    psd /= np.amax(psd)
    psd *= bins**2
    return make_signal(fd=np.sqrt(np.fft.fftshift(psd)), fs=bins*fbin)

# f = np.linspace(-3, 3-1/100, 600)
# msk = calc_msk(f)
# BT = [0.3, 0.5]
# gmsk = []
# gfsk = []
# for W in BT:
#     H_G = np.exp((np.log(2)/-2)*(f/W)**2)
#     gmsk.append(10*np.log10(msk*H_G))
# 
# msk = 10*np.log10(msk)
# 
# #plt.figure(figsize=(7, 4), dpi=300)
# plt.plot(f, msk, label='MSK', color='green')
# plt.plot(f, gmsk[1], label='GMSK BT = 0.5', color='red')
# plt.plot(f, gmsk[0], label='GMSK BT = 0.3', color='black')
# plt.title("Normalized Power Spectral Density")
# plt.xlabel('Frequency Offset / Bit Rate (Hz/bit/s)')
# plt.ylabel('Spectral Power Level (dB)')
# plt.ylim((-100,20))
# plt.legend()
# plt.grid()
# plt.show()# plt.savefig('PSD.png')
