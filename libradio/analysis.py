""" Methods for analyzing data in Signal objects
    Cole Nielsen 2019
"""

import numpy as np
from libradio._signal import *

DB_PER_BIT = 6.02

def measure_in_band_sfdr(signal, bw=1000, verbose=True, *args, **kwargs):
    '''For non-single tone signals: finds SFDR in signal in 0-bandwidth (bw) vs out of band
    '''
    if not any(signal.fd): # if freq domain data not computed
        print("\n* Computing frequency domain data for signal object. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    bw_index = freq_to_index(signal, bw)
    signal_max = 20*np.log10(np.amax(np.abs(signal.fd[:bw_index])))
    noise_max = 20*np.log10(np.amax(np.abs(signal.fd[bw_index:int(signal.samples/2)])))
    sfdr_db = signal_max - noise_max
    if verbose:
        print("\n* In band SFDR = %0.2f dB"%(sfdr_db))
        print("\tBaseband bandwidth = %0.1f Hz"%bw)
        print("\tSignal name = %s"%signal.name)
    return sfdr_db


def measure_sfdr(signal, tone_freq=1000, tone_bw=10, verbose=True, *args, **kwargs):
    '''Measures SFDR for single tone at tone_freq
    '''
    if not any(signal.fd): # if freq domain data not computed
        print("\n* Computing frequency domain data for signal object. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    tone_index = freq_to_index(signal, abs(tone_freq))
    bw_delta = int(round((tone_bw / signal.fbin)/2))
    signal_max = 20*np.log10(np.amax(np.abs(signal.fd[tone_index-bw_delta:tone_index+bw_delta])))
    noise_max = 20*np.log10(np.amax(np.concatenate((np.abs(signal.fd[:tone_index-bw_delta]),
                                                    np.abs(signal.fd[tone_index+bw_delta:int(signal.samples/2)])))))
    sfdr_db = signal_max - noise_max
    if verbose:
        print("\n* SFDR = %0.2f dB"%(sfdr_db))
        print("\tTone frequency = %0.1f Hz, bandwidth = %0.1f"%(tone_freq, tone_bw))
        print("\tSignal name = %s"%signal.name)
    return sfdr_db


def measure_in_band_snr(signal, bw=1000, verbose=True, *args, **kwargs):
    '''For non-single tone signals: finds SNR in signal in 0-bandwidth (bw) vs out of band
    '''
    if not any(signal.fd): # if freq domain data not computed
        print("\n* Computing frequency domain data for signal object. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    bw_index = freq_to_index(signal, bw)
    signal_power = 10*np.log10(np.sum(np.abs(signal.fd[:bw_index])**2))
    noise_power = 10*np.log10(np.sum(np.abs(signal.fd[bw_index:int(signal.samples/2)])**2))
    snr_db = signal_power-noise_power
    n_bits = snr_db/DB_PER_BIT
    if verbose:
        print("\n* In band SNR = %0.2f dB, Effective N bits = %0.2f"%(snr_db,n_bits))
        print("\tBaseband bandwidth = %0.1f Hz"%bw)
        print("\tSignal name = %s"%signal.name)
    return snr_db


def measure_sndr(signal, tone_freq=1000, tone_bw=10, verbose=True, *args, **kwargs):
    '''Measures SNDR for single tone at tone_freq
    '''
    if not any(signal.fd): # if freq domain data not computed
        print("\n* Computing frequency domain data for signal object. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    tone_index = freq_to_index(signal, abs(tone_freq))
    bw_delta = int(round((tone_bw / signal.fbin)/2))
    signal_power = 10*np.log10(np.sum(np.abs(signal.fd[tone_index-bw_delta:tone_index+bw_delta])**2))
    noise_power = (10*np.log10(np.sum(np.abs(signal.fd[:tone_index-bw_delta]))
                               + np.sum(np.abs(signal.fd[tone_index+bw_delta:int(signal.samples/2)])**2)))
    sndr_db = signal_power - noise_power
    n_bits = (sndr_db-1.76) / DB_PER_BIT
    if verbose:
        print("\n* SNDR = %0.2f dB, Effective N bits = %0.2f"%(sndr_db,n_bits))
        print("\tTone frequency = %0.1f Hz, bandwidth = %0.1f"%(tone_freq, tone_bw))
        print("\tSignal name = %s"%signal.name)
    return sndr_db


def measure_rms(signal, verbose=True, *args, **kwargs):
    """ Measures RMS level of signal
    """
    rms = np.std(signal.td)
    if verbose:
        print("\nSignal RMS = %f"%rms)
        print("\tSignal name = %s"%signal.name)
    return rms
