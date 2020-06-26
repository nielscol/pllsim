""" Methods for calcuting things related to phase noise
"""

import numpy as np
from libpll._signal import make_signal
KB = 1.38064852e-23

###############################################################################
# Phase noise estimation for discrete random walk ring oscillator model
###############################################################################

def fom_to_pn(fom, power, freq, df):
    fom_lin = 10**(fom/10)
    return fom_lin*freq**2/(1e3*power*df**2)

def phase_noise_est(k, n, df, tstep, m=0.61):
    return k**2/(n*(2*np.pi*df*tstep*m)**2)

def _min_ro_pn(f0, df, power, temp, fc=0, q=1):
    """ Returns the theorectical pn limit for ring oscillator
        presumably quality factor is 1
    """
    if df == 0: return np.nan
    else: return 7.33*KB*temp*(f0/df)**2/power*(1+fc/df) + 7.33*8*q**2*KB*temp/power
min_ro_pn = np.vectorize(_min_ro_pn, otypes=[float])

def _leeson_pn(f0, df, power, temp, q, nf, fc=0):
    """ Leeson's phase noise model, q is loaded quality factor, nf is amplifier
        noise factor, fc is the flicker noise corner.
    """
    return 0.5*nf*KB*temp/power*(1+(f0/(2*q*df))**2)*(1+fc/df) + F*KB*temp/power
leeson_pn = np.vectorize(_leeson_pn, otypes=[float])

def min_ro_power(pn, fo, df, temp, fc=0, mode="db"):
    """ Returns minimum possible needed to achieve a given phase noise value
    """
    if mode == "db":
        return 10*np.log10(7.33*KB*temp*(f0/df)**2*(fc/df+1)/10**(pn/10))
    else:
        return 7.33*KB*temp*(f0/df)**2*(fc/df+1)/pn


def rw_gain_fom(fom_db, fosc, power, fs):
    s0_osc = 10**(fom_db/10.0)*fosc**2*1e-3/power
    return 2*np.pi*np.sqrt(s0_osc/fs)


def _tdc_pn2(fclk, n, g, tdel):
    return fclk*np.abs(2*np.pi*n*g)**2*tdel**2/12
tdc_pn2 = np.vectorize(_tdc_pn2, otypes=[float])

def tdc_pn(fclk, div_n, tdc_steps, g):
    return np.abs(2*np.pi*div_n*g)**2/(12*tdc_steps**2*fclk)
# tdc_pn = np.vectorize(_tdc_pn, otypes=[float])

def tdc_pn2(fclk, div_n, tdc_steps, tdc_noise_pow, g):
    return np.abs(2*np.pi*div_n*g/tdc_steps)**2*(tdc_noise_pow/fclk)

def bbpd_pn(fclk, div_n, sigma_ph, g):
    return (np.pi/2 - 1)*np.abs(sigma_ph*div_n*g)**2/fclk

def dco_pn(pn_dco, pn_dco_df, g):
    kpn = pn_dco*pn_dco_df**2
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    return ndco
# dco_pn = np.vectorize(dco_pn, otypes=[float])

def dco_quant_pn(fclk, div_n, tdc_steps, lf, g):
    return np.abs(2*np.pi*div_n*g/lf)**2/(12*tdc_steps**2*fclk)

def _div_pn(fclk, div_n, div_jit, g):
    return fclk*np.abs(2*np.pi*div_n*div_jit*g)**2
div_pn = np.vectorize(_div_pn, otypes=[float])


def _pll_tf(f, _type, k, fz, fp, delay):
    """ Closed loop PLL transfer function, uses lumped gain k instead of PLL model parameters
    """
    wp = 2*np.pi*fp
    wz = 2*np.pi*fz
    s = 2j*np.pi*f
    return k*np.exp(-s*delay)*(s/wz + 1)/(s**_type*(s/wp + 1) + k*np.exp(-s*delay)*(s/wz + 1))
pll_tf = np.vectorize(_pll_tf, otypes=[complex])

def _lf_pn(f, fclk, lf_params, kdco, b1, b2):
    c1 = b1 + b2
    c2 = b1 + 2*b2
    return kdco**5/(12*fclk*f**2)*(4)*np.abs((1-pll_tf(f,**lf_params))/(1+c1-2j*np.pi*f*c2))**2
lf_pn = np.vectorize(_lf_pn, otypes=[float])

def make_pn_sig(fbin, fmax, pn_dco, pn_dco_df, fclk, _type, m, n, k, ki, fz, fp,
                delay, fn, bw, damping, a0, a1, b0, b1, b2, label="",
                *args, **kwargs):

    bins = int(round(2*fmax/fbin))
    freqs = fbin*(np.arange(bins)-int(bins/2))

    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)

    kpn = pn_dco*pn_dco_df**2
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0

    # ndco = min_ro_pn(fclk*n, freqs, posc, temp)*np.abs(1-g)**2
    ntdc = tdc_pn(fclk, n, m, g)*np.abs(g)**2

    psd = np.fft.fftshift(ndco+ntdc)*bins**2
    if np.isnan(psd[0]): psd[0] = 0.5*(psd[1] + psd[-1])

    return make_signal(fd=np.sqrt(psd), fs=bins*fbin)
