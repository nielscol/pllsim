import numpy as np
import matplotlib.pyplot as plt


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


def make_signal(td=[], fd=[], fs=None, bits=None, bitrate=None,
                signed=None, name="", autocompute_fd=False, verbose=False,
                force_even_samples=True, *args, **kwargs):
    """Method to assist with creation of Signal objects.
    * Will not automatically compute fd = fft(td) unless autocompute_fd is set. This is to save time
    when not needed.
    """
    if len(td) and type(td) == list:
        if type(td[0]) == complex:
            td = np.array(td, dtype=np.complex)
        elif type(td[0]) == float:
            td = np.array(td, dtype=np.float)
        elif type(td[0]) == int:
            td = np.array(td, dtype=np.int32)
        else:
            raise Exception("time domain argument td of unsupported type. Use int, float or complex")
    if len(fd) and type(fd) != np.ndarray:
        raise Exception("Please use numpy ndarray as argument type for fd")
    if verbose:
        print("\n")
    if not fs:
        print("* No sampling rate fs provided, assuming 1 Hz.")
        fs = 1
    if len(td) and len(fd):
        raise Exception("It is only allowed to set freq. domain (fd) OR time domain (td) data. Both arguments were passed.")
    elif not len(td) and not len(fd):
        raise Exception("No time domain (td) or frequency domain (fd) data passed.")
    if len(td):
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
    elif len(fd):
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
