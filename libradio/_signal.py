""" Contains data structure "Signal" for representing signals, containing waveform data
    and parameters such as sampling frequencies

    Also, contains methods for Signal object creation, and importion of .wav to Signal objects

    Cole Nielsen 2019
"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
#import collections

#Signal = collections.namedtuple("Signal", ["td", "fd", "fs", "samples", "bits", "signed", "fbin", "name"])

class Signal:
    def __init__(self, td, fd, fs, samples, bits,
                 signed, fbin, name, bitrate=None):
        self.td = td
        self.fd = fd
        self.fs = fs
        self.bitrate = bitrate
        self.samples = samples
        self.bits = bits
        self.signed = signed
        self.fbin = fbin
        self.name = name


class EyeData:
    def __init__(self, eye_data, x_range, y_range, x_len, y_len, n_traces):
        self.eye_data = eye_data
        self.h_range = x_range
        self.v_range = y_range
        self.x_len = x_len
        self.y_len = y_len
        self.n_traces = n_traces

def make_signal(td=[], fd=[], fs=None, bits=None, bitrate=None,
                signed=None, name="", autocompute_fd=False, verbose=False,
                force_even_samples=True, *args, **kwargs):
    """Method to assist with creation of Signal objects.
    * Will not automatically compute fd = fft(td) unless autocompute_fd is set. This is to save time
    when not needed.
    """
    if any(td) and type(td) == list:
        if type(td[0]) == complex:
            td = np.array(td, dtype=np.complex)
        elif type(td[0]) == float:
            td = np.array(td, dtype=np.float)
        elif type(td[0]) == int:
            td = np.array(td, dtype=np.int32)
        else:
            raise Exception("time domain argument td of unsupported type. Use int, float or complex")
    if any(fd) and type(fd) != np.ndarray:
        raise Exception("Please use numpy ndarray as argument type for fd")
    if verbose:
        print("\n")
    if not fs:
        print("* No sampling rate fs provided, assuming 1 Hz.")
        fs = 1
    if any(td) and any(fd):
        raise Exception("It is only allowed to set freq. domain (fd) OR time domain (td) data. Both arguments were passed.")
    elif not any(td) and not any(fd):
        raise Exception("No time domain (td) or frequency domain (fd) data passed.")
    if any(td):
        if len(td) % 2 == 1 and force_even_samples: # make even no. samples
            if verbose:
                print("Removing sample to reshape data to even number of samples")
            td = td[:-1]
        samples = len(td)
        fbin = float(fs)/samples
        if autocompute_fd:
            fd = np.fft.fft(td)
        else:
            if verbose:
                print("* Not pre-computing FFT of time domain data to save time in Signal object")
            fd = np.array(np.zeros(len(td)), dtype=np.complex)
    elif any(fd):
        if len(fd) % 2 == 1 and force_even_samples: # make even no. samples
            if verbose:
                print("Removing sample to reshape data to even number of samples")
            fd = fd[:-1]
        samples = len(fd)
        fbin = float(fs)/samples
        td = np.fft.ifft(fd)
    if verbose:
        print("* Named Signal tuple %s instantiated with properties:"%name)
        print("\tSamples = %d, Sampling rate = %d Hz, Bin delta f = %0.2f Hz"%(samples, fs, fbin))
    return Signal(td, fd, fs, samples, bits, signed, fbin, name, bitrate=bitrate)


def wav_to_signal(file_name, autocompute_fd=False, verbose=True, *args, **kwargs):
    """ import .wav file and create Signal object for it
    """
    fs, data = wavfile.read(file_name)
    if data.dtype == np.int16:
        bits = 16
        signed = True
    elif data.dtype == np.int32:
        bits = 32
        signed = True
    elif data.dtype == np.uint8: # unsigned
        bits = 8
        signed = True # change to be signed...
        data += - 2**(bits-1)
    else:
        raise Exception("Unplanned for data type returned for wavfile.read(). Currently only support int16, int32, uint8")
    if verbose:
        print("\n* Read .wav file \"%s\""%file_name)
        print("\tSampling rate = %d Hz, Samples = %d"%(fs, len(data)))
    return make_signal(td=np.array(data, dtype=np.int32), fs=fs, bits=bits,
                       signed=signed, bitrate=fs*bits, name=file_name,
                       autocompute_fd=autocompute_fd, verbose=verbose,
                       *args, **kwargs)


def save_signal_to_wav(signal, file_name, dtype=np.int16, verbose=True, *args, **kwargs):
    if not signal.signed:
        data = signal.td - 2**(signal.bits-1)
    else:
        data = signal.td
    if verbose:
        print("\n* Saving to %s"%file_name)
    wavfile.write(file_name, rate=signal.fs, data=np.array(data, dtype=dtype))


def freq_to_index(signal, freq, verbose=True, *args, **kwargs):
    """ computes index of bin in FFT(Signal.td) corresponding provided frequency
    """
    if not any(signal.fd):
        print("\n* Frequency domain indexing requested, computing frequency domain data first. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    if freq > -0.5*signal.fbin and freq <= signal.fs/2.0 - signal.fbin:
        n = int(round(freq / signal.fbin))
    elif freq > signal.fs/2.0 - signal.fbin:
        n = signal.samples/2 - 1
    elif freq <= -0.5*signal.fbin and freq >= -signal.fs/2.0:
        n = int(round(freq/signal.fbin)) + signal.samples
    else:
        n = signal.samples - 1
    return int(n)


def generate_quantized_tone(tone_freq, fs, samples, bits, signed=True,
                            noise_lsbs=0.0, autocompute_fd=False,
                            verbose=True, *args, **kwargs):
    amplitude = (2**bits - 1)/2.0
    time = np.arange(samples)/float(fs)
    tone = amplitude*np.sin(2*math.pi*tone_freq*time)
    noise = np.random.normal(0.0, noise_lsbs, samples)
    td = np.array(np.rint(tone + noise + amplitude), dtype=np.int32)
    td[td<0] = 0
    td[td>2**bits-1] = 2**bits - 1
    td = td - 2**(bits-1) if signed else amplitude
    if verbose:
        print("\n* Generating quantized sinusoidal signal")
        print("\t* Tone freq = %f,\tfs = %f,\tsamples = %f"%(tone_freq,fs, samples))
        print("\t* Bits per sample = %d,\tsigned = %r,\trms noise in lsbs = %f"%(bits, signed, noise_lsbs))
    return make_signal(td=td, fs=fs, bits=bits, signed=signed, bitrate=fs*bits,
                       name="quantized_tone_%.0fHz_%d_bits"%(tone_freq, bits),
                       autocompute_fd=autocompute_fd, verbose=False, *args, **kwargs)


def generate_random_bitstream(length, bitrate=1, name="", autocompute_fd=False,
                              verbose=True, *args, **kwargs):
    message = np.random.choice([0,1], length)
    return make_signal(td=message, bits=1, fs=bitrate, bitrate=bitrate, name=name,
                       autocompute_fd=autocompute_fd, verbose=False, *args, **kwargs)

