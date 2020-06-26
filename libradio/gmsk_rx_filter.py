""" This is based on the filter design code for GMSK from liquidsdr.org - https://github.com/jgaeddert/liquid-dsp
        Some of the code has changed quite a bit going from c to Python
"""

"""
/*
 * Copyright (c) 2007 - 2015 Joseph Gaeddert
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
"""

from math import sqrt, log, pi, exp, sin
import numpy as np
from copy import copy
from scipy.special import erfc
from libradio._signal import make_signal

SQRTLN2 = sqrt(log(2.0))

#############################################################################
#   GMSK Rx/Tx Filter design
#############################################################################

def gmsk_tx_filter(k, m, bt, fs, dt=0.0, norm=False, autocompute_fd=False,
                   verbose=True, *args, **kwargs):
    """ Design GMSK transmit filter
         k      : samples/symbol
         m      : pulse_span
         bt     : rolloff factor (0 < bt <= 1)
         dt     : fractional sample delay
    """
    if k < 1 :
        raise Exception("error: gmsk_tx_filter(): k must be greater than 0\n")
    elif m < 1:
        raise Exception("error: gmsk_tx_filter(): m must be greater than 0\n")
    elif bt < 0.0 or bt > 1.0:
        raise Exception("error: gmsk_tx_filter(): beta must be in [0,1]\n")

    # derived values
    fir_len = k*m+1

    # compute filter coefficients
    t  = np.arange(fir_len)/float(k) - 0.5*m + dt
    tx_fir = q(2*pi*bt*(t-0.5)/SQRTLN2)-q(2*pi*bt*(t+0.5)/SQRTLN2)

    # normalize filter coefficients such that the filter's
    # integral is pi/2
    if norm:
        tx_fir *= 1.0/float(sum(tx_fir))
    else: # pi/2 scale
        tx_fir *= pi/(2.0*sum(tx_fir))

    return make_signal(td=tx_fir, fs=fs, force_even_samples=False,
                       name="gmsk_tx_fir_bt_%.2f"%bt, autocompute_fd=autocompute_fd, verbose=False)


def gmsk_matched_kaiser_rx_filter(k, m, bt_tx, bt_composite, fs, dt=0.0,
                                  delta=1e-3, autocompute_fd=False, verbose=True,
                                  *args, **kwargs):
    """ Design GMSK receive filter
        k      : samples/symbol
        m      : fir span
        bt     : rolloff factor (0 < beta <= 1)
        dt     : fractional sample delay
    """
    # validate input
    if k < 1:
        raise Exception("error: gmsk_matched_kaiser_rx_filter(): k must be greater than 0\n")
    elif m < 1:
        raise Exception("error: gmsk_matched_kaiser_rx_filter(): m must be greater than 0\n")
    elif bt_tx < 0.0 or bt_tx > 1.0:
        raise exception("error: gmsk_matched_kaiser_rx_filter(): beta must be in [0,1]\n")
    elif bt_composite < 0.0 or bt_composite > 1.0:
        raise exception("error: gmsk_matched_kaiser_rx_filter(): beta must be in [0,1]\n")

    # derived values
    fir_len = k*m+1   # filter length
    # design transmit filter
    tx_fir = gmsk_tx_filter(k, m, bt_tx, fs).td

    # start of Rx filter design procedure
    # create 'prototype' matched filter
    prototype_fir = kaiser_filter_prototype(k, m, bt_composite, 0.0)
    prototype_fir *= pi/(2.0*sum(prototype_fir))

    # create 'gain' filter to improve stop-band rejection
    fc = (0.7 + 0.1*bt_tx) / float(k)
    As = 60.0
    oob_reject_fir = kaiser_filter_design(fir_len, fc, As, 0.0)

    # run ffts
    prototype_fd = np.fft.fft(prototype_fir)
    oob_reject_fd = np.fft.fft(oob_reject_fir)
    tx_fd = np.fft.fft(tx_fir)

    # find minimum of reponses
    tx_fd_min = np.amin(np.abs(tx_fd))
    prototype_fd_min = np.amin(np.abs(prototype_fd))
    oob_reject_fd_min = np.amin(np.abs(oob_reject_fd))

    # compute approximate matched Rx response, removing minima, and add correction factor
    rx_fd = (np.abs(prototype_fd) - prototype_fd_min + delta) / (np.abs(tx_fd) - tx_fd_min + delta)
    # Out of band suppression
    rx_fd *= (np.abs(oob_reject_fd) - oob_reject_fd_min) / (np.abs(oob_reject_fd[0]))
    rx_fir = np.fft.fftshift(np.fft.ifft(rx_fd))
    rx_fir = np.real(rx_fir)*k

    return make_signal(td=rx_fir, fs=fs, force_even_samples=False,
                       name="gmsk_kaiser_matched_rx_fir_bt_tx_%.2f_bt_comp_%.2f"%(bt_tx, bt_composite),
                       autocompute_fd=autocompute_fd, verbose=False)


def kaiser_composite_tx_rx_filter(k, m, bt_tx, bt_composite, fs, dt=0.0, delta=1e-3,
                                  norm=False, autocompute_fd=False, verbose=True, *args, **kwargs):
    """ Filter given by gmsk_matched_kaiser_rx_filter() and gmsk_tx_filter() together
        i.e. Kaiser filer but with some out-of-band supression
        k      : samples/symbol
        m      : span of fir
        bt     : rolloff factor (0 < beta <= 1)
        dt     : fractional sample delay
    """
    # validate input
    if k < 1:
        raise Exception("error: kaiser_composite_rx_tx_filter(): k must be greater than 0\n")
    elif m < 1:
        raise Exception("error: kaiser_composite_rx_tx_filter(): m must be greater than 0\n")
    elif bt_tx < 0.0 or bt_tx > 1.0:
        raise exception("error: kaiser_composite_rx_tx_filter(): beta must be in [0,1]\n")
    elif bt_composite < 0.0 or bt_composite > 1.0:
        raise exception("error: kaiser_composite_rx_tx_filter(): beta must be in [0,1]\n")

    # derived values
    fir_len = k*m+1   # filter length

    # create 'prototype' matched filter
    prototype_fir = kaiser_filter_prototype(k, m, bt_composite, 0.0)
    prototype_fir *= pi/(2.0*sum(prototype_fir))

    # create 'gain' filter to improve stop-band rejection
    fc = (0.7 + 0.1*bt_tx) / float(k)
    As = 60.0
    oob_reject_fir = kaiser_filter_design(fir_len, fc, As, 0.0)

    # run ffts
    prototype_fd = np.fft.fft(prototype_fir)
    oob_reject_fd = np.fft.fft(oob_reject_fir)

    # find minimum of reponses
    prototype_fd_min = np.amin(np.abs(prototype_fd))
    oob_reject_fd_min = np.amin(np.abs(oob_reject_fd))

    # compute approximate matched Rx response, removing minima, and add correction factor
    comp_fd = (np.abs(prototype_fd) - prototype_fd_min + delta)
    # Out of band suppression
    comp_fd *= (np.abs(oob_reject_fd) - oob_reject_fd_min) / (np.abs(oob_reject_fd[0]))
    comp_fir = np.fft.fftshift(np.fft.ifft(comp_fd))
    comp_fir = np.real(comp_fir)*k
    if norm:
        comp_fir *= 1.0/float(sum(comp_fir))

    return make_signal(td=comp_fir, fs=fs, force_even_samples=False,
                       name="kaiser_composite_fir_%.2f_bt_comp_%.2f"%(bt_tx, bt_composite),
                       autocompute_fd=autocompute_fd, verbose=False)


def gmsk_matched_rcos_rx_filter(k, m, bt_tx, bt_composite, fs, dt=0.0, delta=1e-3,
                                autocompute_fd=False, verbose=True, *args, **kwargs):
    """ Design GMSK receive filter for raised cosine
        k      : samples/symbol
        m      : fir span
        bt     : tx rolloff factor (0 < beta <= 1)
        beta   :
        dt     : fractional sample delay
    """
    # validate input
    if k < 1:
        raise Exception("error: gmsk_matched_rcos_rx_filter(): k must be greater than 0\n")
    elif m < 1:
        raise Exception("error: gmsk_matched_rcos_rx_filter(): m must be greater than 0\n")
    elif bt_tx < 0.0 or bt_tx > 1.0:
        raise exception("error: gmsk_matched_rcos_rx_filter(): beta must be in [0,1]\n")
    elif bt_composite < 0.0 or bt_composite > 1.0:
        raise exception("error: gmsk_matched_rcos_rx_filter(): beta must be in [0,1]\n")

    # derived values
    fir_len = k*m+1   # filter length
    # design transmit filter
    tx_fir = gmsk_tx_filter(k, m, bt_tx, fs).td

    # start of Rx filter design procedure
    # create 'prototype' matched filter

    t  = np.arange(fir_len)/float(k) - 0.5*m + dt
    prototype_fir = v_raised_cos(t, 1.0, bt_composite)
    prototype_fir *= pi/(2.0*sum(prototype_fir))
    # create 'gain' filter to improve stop-band rejection
    fc = (0.7 + 0.1*bt_tx) / float(k)
    As = 60.0
    oob_reject_fir = kaiser_filter_design(fir_len, fc, As, 0.0)

    # run ffts
    prototype_fd = np.fft.fft(prototype_fir)
    oob_reject_fd = np.fft.fft(oob_reject_fir)
    tx_fd = np.fft.fft(tx_fir)

    # find minimum of reponses
    tx_fd_min = np.amin(np.abs(tx_fd))
    prototype_fd_min = np.amin(np.abs(prototype_fd))
    oob_reject_fd_min = np.amin(np.abs(oob_reject_fd))

    # compute approximate matched Rx response, removing minima, and add correction factor
    rx_fd = (np.abs(prototype_fd) - prototype_fd_min + delta) / (np.abs(tx_fd) - tx_fd_min + delta)
    # Out of band suppression
    rx_fd *= (np.abs(oob_reject_fd) - oob_reject_fd_min) / (np.abs(oob_reject_fd[0]))
    rx_fir = np.fft.fftshift(np.fft.ifft(rx_fd))
    rx_fir = np.real(rx_fir)*k

    return make_signal(td=rx_fir, fs=fs, force_even_samples=False,
                       name="gmsk_rcos_matched_fir_%.2f_bt_comp_%.2f"%(bt_tx, bt_composite),
                       autocompute_fd=autocompute_fd, verbose=False)


def rcos_composite_tx_rx_filter(k, m, bt_tx, bt_composite, fs, dt=0.0,
                                delta=1e-3, autocompute_fd=False, norm=False,
                                verbose=True, *args, **kwargs):
    """ Raised cosine response including out of band supression
        k      : samples/symbol
        m      : fir span
        bt     : tx rolloff factor (0 < beta <= 1)
        beta   :
        dt     : fractional sample delay
    """
    # validate input
    if k < 1:
        raise Exception("error: rcos_composite_tx_rx_filter(): k must be greater than 0\n")
    elif m < 1:
        raise Exception("error: rcos_composite_tx_rx_filter(): m must be greater than 0\n")
    elif bt_tx < 0.0 or bt_tx > 1.0:
        raise exception("error: rcos_composite_tx_rx_filter(): beta must be in [0,1]\n")
    elif bt_composite < 0.0 or bt_composite > 1.0:
        raise exception("error: rcos_composite_tx_rx_filter(): beta must be in [0,1]\n")

    # derived values
    fir_len = k*m+1   # filter length

    # create 'prototype' matched filter

    t  = np.arange(fir_len)/float(k) - 0.5*m + dt
    prototype_fir = v_raised_cos(t, 1.0, bt_composite)
    prototype_fir *= pi/(2.0*sum(prototype_fir))
    # create 'gain' filter to improve stop-band rejection
    fc = (0.7 + 0.1*bt_tx) / float(k)
    As = 60.0
    oob_reject_fir = kaiser_filter_design(fir_len, fc, As, 0.0)

    # run ffts
    prototype_fd = np.fft.fft(prototype_fir)
    oob_reject_fd = np.fft.fft(oob_reject_fir)

    # find minimum of reponses
    prototype_fd_min = np.amin(np.abs(prototype_fd))
    oob_reject_fd_min = np.amin(np.abs(oob_reject_fd))

    # compute approximate matched Rx response, removing minima, and add correction factor
    comp_fd = (np.abs(prototype_fd) - prototype_fd_min + delta)
    # Out of band suppression
    comp_fd *= (np.abs(oob_reject_fd) - oob_reject_fd_min) / (np.abs(oob_reject_fd[0]))
    comp_fir = np.fft.fftshift(np.fft.ifft(comp_fd))
    comp_fir = np.real(comp_fir)*k
    if norm:
        comp_fir *= 1.0/float(sum(comp_fir))

    return make_signal(td=comp_fir, fs=fs, force_even_samples=False,
                       name="rcos_composite_fir_%.2f_bt_comp_%.2f"%(bt_tx, bt_composite),
                       autocompute_fd=autocompute_fd, verbose=False)


#############################################################################
#   Methods used by gmsk_rx_filter()
#############################################################################

def kaiser_filter_prototype(k, m, beta, mu=0.0):
    """ Design (root-)Nyquist filter from prototype
        type   : filter type
        k      : samples/symbol
        m      : fir span
        beta   : excess bandwidth factor, beta in [0,1]
        mu     : fractional sample delay (also dt)
    """
    # compute filter parameters
    n = k*m + 1   # length
    fc = 0.5 / float(k)        # cut-off frequency
    df = beta / float(k)
    As = estimate_req_filter_As(df,n)   # stop-band attenuation

    return kaiser_filter_design(n, fc, As, mu)


def kaiser_filter_design(n, fc, As, mu):
    # validate inputs
    if mu < -0.5 or mu > 0.5:
        raise Exception("error: kaiser_filter_design(), _mu (%12.4e) out of range [-0.5,0.5]\n"%mu)
    elif fc < 0.0 or fc > 0.5:
        raise Exception("error: kaiser_filter_design(), cutoff frequency (%12.4e) out of range (0, 0.5)\n"%_fc)
    elif n == 0:
        raise Exception("error: kaiser_filter_design(), filter length must be greater than zero\n")

    # choose kaiser beta parameter (approximate)
    beta = kaiser_beta_As(As)
    # sinc prototype
    t = np.arange(n) - (n-1.0)/2.0 + mu
    sinc_fir = v_sinx_x(2.0*fc*t)
    # kaiser window
    kaiser_fir = v_kaiser(np.arange(n),n,beta,mu)

    # return composite filter
    return sinc_fir*kaiser_fir


def estimate_req_filter_As(df, N):
    """ estimate filter stop-band attenuation given
        df     :   transition bandwidth (0 < b < 0.5)
        N      :   filter length
    """
    # run search for stop-band attenuation which gives these results
    As0   = 0.01    # lower bound
    As1   = 200.0   # upper bound
    As_hat = 0.0    # stop-band attenuation estimate
    N_hat = 0.0     # filter length estimate

    # perform simple bisection search
    num_iterations = 20
    for i in range(num_iterations):
        # bisect limits
        As_hat = 0.5*(As1 + As0)
        N_hat = estimate_req_filter_len_kaiser(df, As_hat)
        # update limits
        if N_hat < float(N):
            As0 = As_hat
        else:
            As1 = As_hat

    return As_hat


def estimate_req_filter_len_kaiser(df, As):
    """esimate required filter length given transition bandwidth and
    stop-band attenuation (algorithm from [Vaidyanathan:1993])
        df     :   transition bandwidth (0 < df < 0.5)
        As     :   stop-band attenuation [dB] (As > 0)
    """
    if df > 0.5 or df <= 0.0:
        raise Exception("error: estimate_req_filter_len_kaiser(), invalid bandwidth : %f\n"%df)
        exit(1)
    elif As <= 0.0:
        raise Exception("error: estimate_req_filter_len_kaiser(), invalid stopband level : %f\n"%As)

    # compute filter length estimate
    return (As - 7.95)/(14.26*df)


def kaiser_beta_As(As):
    """ returns the Kaiser window beta factor give the filter's target
        stop-band attenuation (As) [Vaidyanathan:1993]
        As     :   target filter's stop-band attenuation [dB], As > 0
    """
    As = abs(As)
    if As > 50.0:
        beta = 0.1102*(As - 8.7)
    elif As > 21.0:
        beta = 0.5842*(As - 21)**0.4 + 0.07886*(As - 21)
    else:
        beta = 0.0

    return beta


def kaiser(n, N, beta, mu):
    """ Kaiser window [Kaiser:1980]
        _n      :   sample index
        _N      :   window length (samples)
        _beta   :   window taper parameter
        _mu     :   fractional sample offset
    """
    # validate input
    if n > N:
        raise Exception("error: kaiser(), sample index must not exceed window length\n")
    elif beta < 0:
        raise Exception("error: kaiser(), beta must be greater than or equal to zero\n")
    elif mu < -0.5 or mu > 0.5:
        raise Exception("error: kaiser(), fractional sample offset must be in [-0.5,0.5]\n")

    t = float(n) - float(N-1)/2.0 + mu
    r = 2.0*t/float(N)
    a = besseli0(beta*sqrt(1-r**2))
    b = besseli0(beta)
    return a / b

v_kaiser = np.vectorize(kaiser, otypes=[float])

def lngamma(z):
    if z < 0:
        raise Exception("error: gammaf(), undefined for z <= 0\n")
    elif z < 10.0:
        # Use recursive formula:
        #    gamma(z+1) = z * gamma(z)
        # therefore:
        #    log(Gamma(z)) = log(gamma(z+1)) - ln(z)
        return lngamma(z + 1.0) - log(z)
    else:
        # high value approximation
        g = 0.5*(log(2*pi)-log(z) )
        g += z*(log(z+(1/(12.0*z-0.1/float(z))))-1)
    return g


# I_0(z) : Modified bessel function of the first kind (order zero)
NUM_BESSELI0_ITERATIONS = 32
def besseli0(z):
    # TODO : use better low-signal approximation
    if z == 0.0:
        return 1.0
    y = 0.0
    for k in range(NUM_BESSELI0_ITERATIONS):
        t = k * log(0.5*z) - lngamma(float(k) + 1.0)
        y += exp(2*t)

    return y


def q(x):
    return 0.5*erfc(x/sqrt(2))


def sinx_x(x):
    return 1.0 if x==0 else sin(pi*x)/(pi*x)

v_sinx_x = np.vectorize(sinx_x, otypes=[float])


def raised_cos(t, tbit, rolloff):
    tbit=float(tbit)
    if rolloff != 0.0 and abs(tbit/(2.0*rolloff)) == t:
        return (pi/(4.0*tbit))*sinx_x(1/(2.0*rolloff))
    elif (2*rolloff*t/tbit)**2 == 1.0:
        if raised_cos(t*(1+1e-6), tbit,rolloff) < 0.25:
            return 0.0
        else:
            return 0.5
    else:
        return (1.0/tbit)*sinx_x(t/tbit)*np.cos(pi*rolloff*t/tbit)/(1.0-(2*rolloff*t/tbit)**2)

v_raised_cos = np.vectorize(raised_cos, otypes=[float])

#import matplotlib.pyplot as plt
#def gmsk_pulse(t, bt, tbit):
#    return (1/(2.0*tbit))*(q(2*pi*bt*(t-0.5*tbit)/SQRTLN2)-q(2*pi*bt*(t+0.5*tbit)/SQRTLN2))
#v_gmsk_pulse = np.vectorize(gmsk_pulse, otypes=[float])

#oversampling=16
#pulse_span=64
#bt=0.3
# Make GMSK pulse shape
#pulse_len = int(oversampling*pulse_span)
#t = (np.arange(pulse_len)-pulse_len/2)/float(oversampling)
#pulse_shape = 2*v_gmsk_pulse(t, bt, 1.0)
# add an offset to pulse so sum of samples = 1.0
#pulse_error = np.sum(pulse_shape)-oversampling
#pulse_shape *= pi/(2.0*sum(pulse_shape))
#tx_filt = gmsk_tx_filter(16, pulse_span, 0.3, 0.0)
#rx_filt = gmsk_rx_filter(16, pulse_span, 0.3, 0.0)
#composite = np.convolve(tx_filt, rx_filt, mode="full")
#plt.plot(tx_filt)
#plt.plot(rx_filt)
#plt.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(composite)))))
#plt.show()
#print(sum(pulse_shape), sum(tx_filt))
#plt.plot(pulse_shape, label="_tx")
#plt.plot(rx_filt, label="rx")
#plt.plot(tx_filt, label="tx")
#plt.legend()
#plt.show()
#plt.plot(20.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx_filt)))))
#plt.show()
