""" Methods for synchronization related things
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from libradio.tools import timer
from libradio._signal import make_signal
import json
from copy import copy

####################################################################################################
#   Methods for getting codes with low autocorrelation sidelobes
####################################################################################################

def get_precomputed_codes():
    '''Precomputed codes for N=1 to N=24
    '''
    print("\n* Reading lib/sync_codes.json to obtain synchronization codes dictionary")
    f = open("./sync_codes.json","r")
    data = f.read()
    f.close()
    return {int(k):v for (k,v) in json.loads(data).items()}


def code_string(code):
    s = ""
    for x in code:
        if x == 0:
            s += "0"
        else:
            s += "+" if x > 0.0 else "-"
    return s


def gen_array_from_num(num, bits):
    arr = np.zeros(bits)
    for n in range(bits):
        arr[n] = 1 if (2**n & num)>>n else -1
    return arr


def lagged_corr_power(autocorrelation,bits):
    n = bits
    return np.sum(autocorrelation[:n-1]**2) + np.sum(autocorrelation[n:]**2)

v_lagged_corr_power = np.vectorize(lagged_corr_power, otypes=[np.int64])


@timer
def find_code(bits, verbose=True, *args, **kwargs):
    if verbose:
        print("Finding sequency of %d bits with minimum sideband autocorrelation power"%bits)
    best_code = None
    best_cost = np.inf
    for n in range(2**bits):
        #code = np.random.choice([-1,1], size=24)
        code = gen_array_from_num(n, bits=bits)
        autocorrelation = np.correlate(code, code, mode="full")
        cost = lagged_corr_power(autocorrelation, bits=bits)
        if cost < best_cost and any(code): # abs(sum(code))<=1:
            best_code = code
            best_cost = cost
            if verbose:
                print("\t├─ NEW BEST CODE, n=%d"%n)
                print("\t│\t├─ Code = %s"%code_string(code))
                #print(ac_code)
                #print(np.argmax(ac_code))
                print("\t│\t└─ Correlation sideband power = %.1f"%lagged_corr_power(autocorrelation, bits=bits))
    if verbose:
        print("\t│")
        print("\t└─ DONE. Optimal %d bits code = %s"%(bits, code_string(best_code)))
    #plt.plot(np.correlate(best_code, best_code, mode="full"))
    #plt.show()
    return best_code

####################################################################################################
#   Methods for synchronization in waveforms
####################################################################################################

def make_sync_fir(sync_code, pulse_fir, oversampling, autocompute_fd=False, verbose=True,
                 *args, **kwargs):
    """ Takes binary sync code word, oversamples it and convolves it with a pulse
        shape finite impulse response to make a FIR sequence that can be used on
        Rx signal for synchronization.
    """
    _sync_code = np.array(sync_code, dtype=float)
    _sync_code[_sync_code<=0] = -1.0
    _sync_code[_sync_code>0] = 1.0
    sync_fir = np.zeros(len(_sync_code)*oversampling)
    sync_fir[np.arange(len(_sync_code))*oversampling] = _sync_code
    sync_fir = np.convolve(sync_fir, pulse_fir.td, mode="full")
    return make_signal(td=sync_fir, fs=pulse_fir.fs, name="sync_code_"+pulse_fir.name, autocompute_fd=autocompute_fd, verbose=False)


def frame_data(signal, sync_code, payload_len, fs, bitrate, sync_pos="center", autocompute_fd=False,
              verbose=True, *args, **kwargs):
    """ Takes data and creates with frames with data payload and sync field
        If sync_pos is "start":
        |<-sync->|<---------------payload----------------->|
        If sync_pos is "center":
        |<------payload----->|<-sync->|<------paylod------>|

        Will zero pad if not enough data passed to fill all frame
    """
    sync_code = np.array(sync_code)
    n_frames = int(np.ceil(len(signal.td)/payload_len))
    f_len = payload_len+len(sync_code)
    s_len = len(sync_code)
    p_len = payload_len
    message = np.zeros(int(n_frames*p_len))
    message[:len(signal.td)] = signal.td
    td = np.zeros(int(n_frames*f_len))
    c_offset = int(payload_len/2.0)
    for n in range(n_frames):
        if sync_pos == "center":
            td[n*f_len:n*f_len+c_offset] = message[n*p_len:n*p_len+c_offset]
            td[n*f_len+c_offset:n*f_len+c_offset+s_len] = sync_code
            td[n*f_len+c_offset+s_len:n*f_len+f_len] = message[n*p_len+c_offset:n*p_len+p_len]
        elif sync_pos == "start":
            td[n*f_len:n*f_len+s_len] = sync_code
            td[n*f_len+s_len:n*f_len+f_len] = message[n*p_len:n*p_len+p_len]

    return make_signal(td=td, fs=fs, bitrate=bitrate, name=signal.name+"_%db_frames_%db_sync"%(f_len, s_len),
                       autocompute_fd=autocompute_fd, verbose=False)


def frame_data_bursts(signal, sync_code, payload_len, fs, bitrate, fps, sync_pos="center", autocompute_fd=False,
              verbose=True, *args, **kwargs):
    """ Takes data and creates with frames with data payload and sync field
        Bursts evenly spaced with specified bitrate, but symbol rate increased to fs
        If sync_pos is "start":
        |<-sync->|<---------------payload----------------->|
        If sync_pos is "center":
        |<------payload----->|<-sync->|<------paylod------>|

        Will zero pad if not enough data passed to fill all frame
    """
    sync_code = np.array(sync_code)
    n_frames = int(np.ceil(len(signal.td)/payload_len))
    f_len = payload_len+len(sync_code) + int((fs-bitrate)/float(fps))
    f_len_error = (fs - bitrate)/float(fps) - int((fs-bitrate)/float(fps))
    s_len = len(sync_code)
    p_len = payload_len
    message = np.zeros(int(n_frames*p_len))
    message[:len(signal.td)] = signal.td
    td = np.zeros(int(n_frames*(f_len+f_len_error)))
    c_offset = int(payload_len/2.0)
    for n in range(n_frames):
        error = int(n*f_len_error)
        if sync_pos == "center":
            td[error+n*f_len:error+n*f_len+c_offset] = message[n*p_len:n*p_len+c_offset]
            td[error+n*f_len+c_offset:error+n*f_len+c_offset+s_len] = sync_code
            td[error+n*f_len+c_offset+s_len:error+n*f_len+s_len+p_len] = message[n*p_len+c_offset:n*p_len+p_len]
        elif sync_pos == "start":
            td[n*f_len:n*f_len+s_len] = sync_code
            td[n*f_len+s_len:n*f_len+f_len] = message[n*p_len:n*p_len+p_len]

    return make_signal(td=td, fs=fs, bitrate=bitrate, name=signal.name+"_%db_frames_%db_sync"%(f_len, s_len),
                       autocompute_fd=autocompute_fd, verbose=False)


def detect_sync(signal, sync_code, payload_len, oversampling):
    f_len = oversampling*(len(sync_code)+payload_len)
    n_frames = int(np.floor(len(signal.td)/float(f_len)))
    peak_indices = []
    peak_values = []
    for n in range(n_frames):
        _slice = signal.td[n*f_len:n*f_len+f_len]
        peak_indices.append(np.argmax(np.abs(_slice))+n*f_len)
        peak_values.append(signal.td[peak_indices[-1]])
    return peak_indices, peak_values

