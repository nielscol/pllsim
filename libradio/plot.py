""" Methods for plotting data in Signal objects
    Cole Nielsen 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libradio.tools import *
from libradio.clock_recovery import *
from libradio.sync import detect_sync
from libradio._signal import EyeData
import math
from copy import copy
from scipy import ndimage
from scipy.stats import norm
from multiprocessing import Pool

###################################################################################
#   BASIC TIME DOMAIN / POWER SPECTRUM PLOTTING
###################################################################################

def plot_td(signal, verbose=True, label="", title="", alpha=1.0, *args, **kwargs):
    """ Plots time domain data for signal
    """
    if verbose:
        print("\n* Plotting signal in time domain")
        print("\tSignal.name = %s"%signal.name)
    times = np.arange(signal.samples)/float(signal.fs)
    plt.ylabel("Signal")
    plt.xlabel("Time [s]")
    plt.grid()
    if label != "":
        plt.plot(times, signal.td, label=label, alpha=alpha)
        plt.legend()
    else:
        plt.plot(times, signal.td, label=signal.name, alpha=alpha)
    plt.title("Time domain "+title)

def plot_taps(signal, oversampling=1, verbose=True, label="", title="", alpha=1.0, *args, **kwargs):
    """ Plots time domain data for signal
    """
    if verbose:
        print("\n* Plotting signal in time domain")
        print("\tSignal.name = %s"%signal.name)
    times = np.arange(len(signal.td))/float(oversampling)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [Symbols]")
    plt.grid()
    plt.title("Time domain "+title)

    plt.scatter(times, signal.td, color="C0")
    for n, point in enumerate(signal.td):
        plt.plot([times[n], times[n]],[0, point], color="C0")

def plot_fd(signal, log=True, label="", title="", alpha=1.0, verbose=True, *args, **kwargs):
    """ Plots spectral data for signal
    """
    if not any(signal.fd): # freq. domain not calculated
        print("\n* Calculating frequency domain representation of signal. This may be slow...")
        print("\tSignal.name = %s"%signal.name)
        signal.fd = np.fft.fft(signal.td)
    if verbose:
        print("\n* Plotting signal %s in frequency domain"%signal.name)
    if len(signal.td)%2==0:
        freqs = (np.arange(signal.samples) - (signal.samples/2)) * signal.fbin
    else:
        freqs = (np.arange(signal.samples) - ((signal.samples-1)/2)) * signal.fbin
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    if log:
        if verbose:
            print("\tFFT - Log [dB] scale")
        plt.ylabel("FFT(Signal) [dB]")
        plt.plot(freqs, 20*np.log10(np.abs(np.fft.fftshift(signal.fd))), label=label, alpha=alpha)
    else:
        if verbose:
            print("\tFFT - Magnitude")
        plt.ylabel("FFT(Signal) [magnitude]")
        plt.plot(freqs, np.abs(np.fft.fftshift(signal.fd)), label = label, alpha=alpha)
    if label != "":
        plt.legend()
    plt.title("Power Spectral Density "+title)
    plt.legend()


def plot_histogram(signal, bins=100, fit_normal=False, orientation="vertical", ax_label="[U]",
                   label="", title="", alpha=1.0, verbose=True, *args, **kwargs):
    """ Plots histogram for dependent variable of time series
    """
    if verbose:
        print("\n* Plotting signal in time domain")
        print("\tSignal.name = %s"%signal.name)
    hist = plt.hist(signal.td, bins=bins, density=True, alpha=alpha, label=label,
                    orientation=orientation)
    if orientation == "vertical":
        plt.xlabel(ax_label)
    elif orientation == "horizontal":
        plt.ylabel(ax_label)
    plt.ylabel("Density")
    plt.title("Histogram "+title)
    if fit_normal:
        hist = np.histogram(signal.td, bins=bins, density=True)
        mean = np.mean(signal.td)
        stdev = np.std(signal.td)
        if orientation == "vertical":
            x = np.linspace(np.amin(signal.td), np.amax(signal.td), bins)
            y = norm.pdf(x, mean, stdev)
        elif orientation == "horizontal":
            y = np.linspace(np.amin(signal.td), np.amax(signal.td), bins)
            x = norm.pdf(y, mean, stdev)
        #s = label
        s = label + "\n" if label != "" else ""
        plt.plot(x, y, label="%smu=%f\nstdev=%f"%(s,mean,stdev), alpha=alpha)
        #plt.plot(x, y, label="%s"%(s), alpha=alpha)
    if label != "" or fit_normal:
        plt.legend()

    return hist

def plot_sync_detect_bellcurve(correlation, sync_code, payload_len, oversampling,
                               orientation="vertical", ax_label="[U]", label="",
                               title="", alpha=1.0, verbose=True, *args, **kwargs):
    indices, vals = detect_sync(correlation, sync_code, payload_len, oversampling)

    mean = np.mean(vals)
    stdev = np.std(vals)
    if orientation == "vertical":
        x = np.linspace(mean-5.0*stdev, mean+5.0*stdev, 51)
        y = norm.pdf(x, mean, stdev)/norm.pdf(mean, mean, stdev)
    elif orientation == "horizontal":
        y = np.linspace(mean-5.0*stdev, mean+5.0*stdev, 51)
        x = norm.pdf(y, mean, stdev)/norm.pdf(mean, mean, stdev)
    #s = label
    s = label + "\n" if label != "" else ""
    plt.plot(x, y, label="%smu=%f\nstdev=%f"%(s,mean,stdev), alpha=alpha)
    #plt.plot(x, y, label="%s"%(s), alpha=alpha)
    if label != "" or fit_normal:
        plt.legend()

###################################################################################
#   IQ PHASE / MAGNITUDE
###################################################################################

def plot_iq_phase_mag(i, q, verbose=True, label="", title="", *args, **kwargs):
    """ Plots IQ phase and magnitude versus time
    """
    if verbose:
        print("\n* Plotting IQ signal phase and magnitude")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    times = np.arange(i.samples)/float(i.fs)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.plot(times, np.arctan2(q.td, i.td), label=label+" Phase")
    plt.plot(times, np.hypot(i.td, q.td), label=label+" Magnitude")
    plt.title("IQ Phase and Magnitude "+title)
    plt.legend()


def plot_phase_histogram(i, q, verbose=True, label="", title="", *args, **kwargs):
    """ Plots histogram of IQ phase
    """
    if verbose:
        print("\n* Plotting IQ signal phase histrogram")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    plt.hist(np.arctan2(q.td, i.td), bins=128, density=True, label=label)
    plt.xlabel("IQ Phase")
    plt.ylabel("Density")
    plt.title("IQ Phase Histrogram "+title)
    if label != "":
        plt.legend()

###################################################################################
#   IQ CONSTELLATION
###################################################################################

def plot_constellation(i, q, verbose=True, label="", title="", *args, **kwargs):
    """ Plots IQ constellation with lines
    """
    if verbose:
        print("\n* Plotting IQ signal constellation")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    plt.plot(i.td, q.td, label=label)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.grid()
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("IQ Constellation"+title)
    if label != "":
        plt.legend()


def plot_constellation_density(i, q, log=True, _3d=False, ax_dim=250, cmap="inferno",
                               label="", title="", verbose=True, force_ax_len=None, *args, **kwargs):
    """ Plots IQ constellation with intensity grading (density)
    """
    if verbose:
        print("\n* Plotting IQ signal constellation")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    max_r = np.amax(np.hypot(i.td,q.td))

    # eye parameters
    uis_in_waveform = 3.0
    uis_in_plot = 2
    plot_aspect_ratio = 2
    r_padding = 1.25
    if force_ax_len:
        max_r = force_ax_len/r_padding

    raster_height = raster_width = ax_dim

    im = np.zeros((raster_width, raster_height))

    for n, ii in enumerate(i.td[:-1]):
        x0, y0 = float_to_raster_index(i.td[n], q.td[n], -max_r*r_padding, -max_r*r_padding,
                                       max_r*r_padding, max_r*r_padding, raster_height, raster_width)
        x1, y1 = float_to_raster_index(i.td[n+1], q.td[n+1], -max_r*r_padding, -max_r*r_padding,
                                       max_r*r_padding, max_r*r_padding,raster_height, raster_width)
        plot_raster_line(x0, y0, x1, y1, im)
    # apply log scaling to data 
    _im = copy(im)
    if log is True:
        im +=1
        im = np.log(im)
    if _3d:
        x = max_r*r_padding*(np.arange(raster_width)/float(raster_width) - 0.5)
        y = max_r*r_padding*(np.arange(raster_height)/float(raster_height) - 0.5)
        xx, yy = np.meshgrid(x,y)

        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        ha.plot_surface(xx, yy, _im.T[::-1,:], cmap="inferno")
    else:
        plt.imshow(im.T[::-1,:], extent=[-r_padding*max_r,r_padding*max_r,-r_padding*max_r,
                                         r_padding*max_r], cmap=cmap, interpolation='gaussian')
        ax = plt.gca()
        ax.set_aspect(1.0)

    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("IQ Constellation "+title)

    return _im

###################################################################################
#   EYE DIAGRAM
###################################################################################

@timer
def plot_eye_lines(signal, bits_per_sym = 1, interp_factor=10, interp_span=128,
                   remove_ends=100, recovery="constant_f", est_const_f=False,
                   title="", verbose=True, *args, **kwargs): # "constant_f" 
    """ Plots eye diagram with lines
    """
    if verbose:
        print("\n* Plotting Eye Diagram with lines")
        print("\tSignal.name = %s"%signal.name)
    td = signal.td[remove_ends:]
    td = td[:-remove_ends]
    interpolated = sinx_x_interp(td, interp_factor, interp_span)
    ui_samples = interp_factor*signal.fs*bits_per_sym/float(signal.bitrate)
    if recovery in [None, "edge_triggered"]:
        times, slices = slice_edge_triggered(interpolated, ui_samples)
    elif recovery == "constant_f":
        times, slices = slice_constant_f(interpolated, ui_samples, est_const_f)
    elif recovery == "pll_second_order":
        times, slices = slice_pll_so(interpolated, ui_samples)
    for n, t in enumerate(times):
        plt.plot(t, slices[n])
    plt.xlabel("Time [UI]")
    plt.ylabel("Signal")
    plt.title("Eye Diagram "+title)
    plt.xlim((-0.5,1.5))


@timer
def plot_eye_density(signal, eye_vpp=None, raster_height = 500, _3d=False, log=True,
                     bits_per_sym=1, interp_factor=10, interp_span=128, remove_ends=100,
                     recovery="constant_f", est_const_f=False, title="", pools=None,
                     cmap="inferno", verbose=True, sample_lines=False, oversampling=None,
                     plot=True, previous_data=None, sync_code=None, pulse_fir=None,
                     payload_len= None, sync_pos="center", thresh=None, fir_span=None,
                     aspect=2.0, *args, **kwargs):
    """ Plots eye diagram as intensity graded (density)
    """
    if verbose:
        print("\n* Plotting Eye Diagram Density")
        print("\tSignal.name = %s"%signal.name)
        if _3d:
            print("\t3D plotting enabled")
    # interpolate signal, recover clock and segment data into eye sweeps
    _signal = copy(signal)
    _signal.td = _signal.td[remove_ends:]
    _signal.td = _signal.td[:-remove_ends]
    td = _signal.td
    interpolated = sinx_x_interp(td, interp_factor, interp_span)
    ui_samples = interp_factor*signal.fs*bits_per_sym/float(signal.bitrate)
    if recovery in [None, "edge_triggered"]:
        times, slices = slice_edge_triggered(interpolated, ui_samples)
    elif recovery == "constant_f":
        times, slices = slice_constant_f(interpolated, ui_samples, est_const_f)
    elif recovery == "pll_second_order":
        times, slices = slice_pll_so(interpolated, ui_samples)
    elif recovery == "frame_sync":
        times, slices = slice_frame_sync(_signal, interpolated, sync_code, pulse_fir,
                                         payload_len, oversampling, interp_factor,
                                         fir_span, sync_pos, thresh)
    # auto set vertical scale if not given in arguments
    _min = 0.0
    _max = 0.0
    if eye_vpp is None:
        for _slice in slices:
            s_min = np.amin(_slice)
            s_max = np.amax(_slice)
            if s_min < _min:
                _min = s_min
            if s_max > _max:
                _max = s_max
        eye_vpp = (_max-_min)

    n_slices = len(slices)

    # eye parameters
    uis_in_waveform = 3.0
    uis_in_plot = 2
    plot_aspect_ratio = aspect
    y_padding = 1.25

    # eye_vpp, eye_height = self.eye_height()

    plot_aspect_ratio = (uis_in_plot/eye_vpp)/plot_aspect_ratio # correct aspect ratio to the way pyplot expects it (scaling factor)

    raster_height = raster_height
    raster_width = int(raster_height*uis_in_waveform)

    eye_raster = np.zeros((raster_width, raster_height))
    if previous_data:
        eye_raster += previous_data

    # run on multiple processes, will be quicker for large data sets
    if pools != None:
        print("\tRasterizing eye diagram with %d processes ..."%pools)
        #update globals
        samples_per_pool = int(len(times)/float(pools))
        segments = [] # break data into segements for pool
        for n in range(pools):
            t = times[n*samples_per_pool:(n+1)*samples_per_pool]
            s = slices[n*samples_per_pool:(n+1)*samples_per_pool]
            args = dict(times=t, slices=s, raster_height=raster_height, raster_width=raster_width,
                        uis_in_waveform=uis_in_waveform, eye_vpp=eye_vpp, y_padding=y_padding)
            segments.append(args)

        p = Pool(pools)
        data = p.map(pool_rasterize, segments)
        p.terminate()
        for sub_raster in data:
            eye_raster += sub_raster
    else:
        for m, t in enumerate(times):
            for n, sample in enumerate(slices[m][:-1]):
                x0, y0 = float_to_raster_index(t[n], slices[m][n], -0.5*(uis_in_waveform-1.0), -0.5*y_padding*eye_vpp,
                                               0.5*uis_in_waveform+0.5, 0.5*y_padding*eye_vpp, raster_height, raster_width)
                x1, y1 = float_to_raster_index(t[n+1], slices[m][n+1], -0.5*(uis_in_waveform-1.0), -0.5*y_padding*eye_vpp,
                                               0.5*uis_in_waveform+0.5, 0.5*y_padding*eye_vpp, raster_height, raster_width)
                plot_raster_line(x0, y0, x1, y1, eye_raster)

    # apply log scaling to data 
    _eye_raster = copy(eye_raster)
    if plot:
        if log is True:
            eye_raster +=1
            eye_raster = np.log(eye_raster)
        if _3d:
            _raster_width = int(raster_height*uis_in_plot)
            x = (2*np.arange(_raster_width)/float(_raster_width) - 0.5)
            y = eye_vpp*(np.arange(raster_height)/float(raster_height) - 0.5)
            xx, yy = np.meshgrid(x,y)

            hf = plt.figure()
            ha = hf.add_subplot(111, projection='3d')
            cutoff = (uis_in_waveform-uis_in_plot)*raster_height
            __eye_raster = eye_raster[int(cutoff/2):-int(cutoff/2),:]
            __eye_raster = ndimage.gaussian_filter(__eye_raster, 2)
            ha.plot_surface(xx, yy, __eye_raster.T[::-1,:], cmap=cmap)
        else:
            plt.imshow(eye_raster.T[::-1,:], aspect = plot_aspect_ratio,
                       extent=[-0.5*(uis_in_waveform-1.0),0.5*uis_in_waveform+0.5,
                               -0.5*y_padding*eye_vpp,0.5*y_padding*eye_vpp],
                       cmap=cmap, interpolation='gaussian')
            if sample_lines and oversampling:
                for n in range(oversampling):
                    plt.axvline(n/float(oversampling), color="w")

        plt.xlabel("Time [UI]")
        plt.ylabel("Signal")
        plt.title("Eye Diagram (Density) "+title)
        plt.xlim([-0.5*uis_in_plot + 0.5,uis_in_plot*0.5 + 0.5])

    eye = EyeData(_eye_raster, (-0.5*(uis_in_waveform-1.0), 0.5*uis_in_waveform+0.5),
                  (-0.5*y_padding*eye_vpp, 0.5*y_padding*eye_vpp), raster_width,
                  n_slices, raster_height)

    return eye


def pool_rasterize(args):
    return rasterize(**args)


def rasterize(times, slices, raster_height, raster_width, uis_in_waveform, y_padding, eye_vpp):
    eye_raster = np.zeros((raster_width, raster_height))
    for m, t in enumerate(times):
        for n, sample in enumerate(slices[m][:-1]):
            x0, y0 = float_to_raster_index(t[n], slices[m][n], -0.5*(uis_in_waveform-1.0), -0.5*y_padding*eye_vpp,
                                           0.5*uis_in_waveform+0.5, 0.5*y_padding*eye_vpp, raster_height, raster_width)
            x1, y1 = float_to_raster_index(t[n+1], slices[m][n+1], -0.5*(uis_in_waveform-1.0), -0.5*y_padding*eye_vpp,
                                           0.5*uis_in_waveform+0.5, 0.5*y_padding*eye_vpp, raster_height, raster_width)
            plot_raster_line(x0, y0, x1, y1, eye_raster)
    return eye_raster


###################################################################################
#   Jitter and TIE plotting
###################################################################################

def plot_tie(signal, bits_per_sym = 1, alpha=1.0, interp_factor=10, interp_span=128, remove_ends=100,
             recovery="constant_f", est_const_f=False, sync_code=None, pulse_fir=None, payload_len=None,
             sync_pos="center", fir_span=None, oversampling=None, label="", title="", verbose=True,
             *args, **kwargs):
    if verbose:
        print("\n* Plotting Total Interval Error (TIE) trend.")
        print("\tSignal.name = %s"%signal.name)
    tie = get_tie(signal, bits_per_sym, interp_factor, interp_span, remove_ends, recovery,
                  est_const_f, sync_code, pulse_fir, payload_len, sync_pos, fir_span, oversampling)
    t = np.arange(len(tie))*bits_per_sym/float(signal.bitrate)
    plt.plot(t, tie, label=label, alpha=alpha)
    plt.title("Clock-Data Jitter Total Interval Error (TIE) "+title)
    plt.xlabel("Time [s]")
    plt.ylabel("TIE [UI]")
    if label != "":
        plt.legend()

def plot_jitter_spectrum(signal, bits_per_sym = 1, alpha=1.0, interp_factor=10, interp_span=128, remove_ends=100,
             recovery="constant_f", est_const_f=False, sync_code=None, pulse_fir=None, payload_len=None,
             sync_pos="center", fir_span=None, oversampling=None, label="", title="", verbose=True,
                         *args, **kwargs):
    if verbose:
        print("\n* Plotting Jitter Spectum.")
        print("\tSignal.name = %s"%signal.name)
    tie = get_tie(signal, bits_per_sym, interp_factor, interp_span, remove_ends, recovery,
                  est_const_f, sync_code, pulse_fir, payload_len, sync_pos, fir_span, oversampling)
    samples = len(tie)
    if samples%2 == 1: #odd
        tie = tie[:-1]
        samples -= 1

    f = (np.arange(samples) - samples/2.0)*(signal.bitrate/float(bits_per_sym*samples))
    jitter_spectrum = 20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(tie))))
    plt.plot(f, jitter_spectrum, label=label, alpha=alpha)
    plt.title("Clock-Data Jitter Spectrum "+title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Spectral Density [s^2/Hz]")
    if label != "":
        plt.legend()


def plot_jitter_histogram(signal, bins=100, alpha=1.0, bits_per_sym = 1, interp_factor=10,
                          interp_span=128, remove_ends=100, recovery="constant_f", est_const_f=False,
                          sync_code=None, pulse_fir=None, payload_len=None, sync_pos="center",
                          fir_span=None, oversampling=None, label="", title="", verbose=True,
                          *args, **kwargs):
    if verbose:
        print("\n* Plotting Jitter Histogram.")
        print("\tSignal.name = %s"%signal.name)
    tie = get_tie(signal, bits_per_sym, interp_factor, interp_span, remove_ends, recovery,
                  est_const_f, sync_code, pulse_fir, payload_len, sync_pos, fir_span, oversampling)
    plt.hist(tie, bins=bins, density=True, label=label, alpha=alpha)
    plt.title("Clock-Data Jitter Distribution "+title)
    plt.xlabel("Time [UI]")
    plt.ylabel("Density")
    if label != "":
        plt.legend()

###################################################################################
#   Auxiliary methods used my main plotting methods
###################################################################################

def iq_to_coordinate(i, q, ax_dim):
    bin_size = 2.0/float(ax_dim)
    ii = int(round((i + 1.0)/bin_size))
    qq = int(round((q + 1.0)/bin_size))
    ii = ax_dim - 1 if ii >= ax_dim else ii
    qq = ax_dim - 1 if qq >= ax_dim else qq
    ii = 0 if ii < 0 else ii
    qq = 0 if qq < 0 else qq
    return (ii,qq)


@timer
def slice_edge_triggered(td, ui_samples):
    """ Slices waveform in segments at each zero crossing and plots
    """
    crossings = crossing_times(td)
    times, slices = segment_data(td, crossings, ui_samples)
    return times, slices


@timer
def slice_frame_sync(signal, interpolated, sync_code, pulse_fir, payload_len,
                     oversampling, interp_factor, fir_span, sync_pos, thresh=None):
    """ Slices data based off of correlation to sync pattern in framed data
    """
    # get crossings UNINTERPOLATED
    crossings = frame_sync_recovery(signal, sync_code, pulse_fir, payload_len,
                                    oversampling, fir_span, sync_pos, thresh)
    # multiply crossings with oversampling so it corresponds to interpolated waveform
    crossings_upsampled = crossings*interp_factor
    times, slices = segment_data(interpolated, crossings_upsampled, oversampling*interp_factor)
    return times, slices

@timer
def slice_constant_f(td, ui_samples, est_const_f=True):
    """ Assumes clock is constant frequency, attempts to recover that clock by estimating phase
        and period from waveform crossings, then slices waveform at recovered clock crossings and plots
    """
    clk_crossings, clk_period, clk_phase = constant_f_recovery(td, ui_samples, est_const_f)
    # segment data for eye
    times, slices = segment_data(td, clk_crossings, ui_samples)
    return times, slices


@timer
def segment_data(td, crossings, ui_samples, offset=0.0):
    """ Chops up data into 3 ui slices for each crossing with 1 UI before the crossing and 2 after
        Also returns times normalized to UIs for each signal slice
    """
    td_len = len(td)
    _ui_samples = round(ui_samples)
    times = []
    slices = []
    for crossing in crossings:
        if round(crossing) > _ui_samples and round(crossing) < td_len - 2*_ui_samples:
            _slice = td[int(round(crossing)-_ui_samples):int(round(crossing)+2*_ui_samples)]
            _time = np.arange(len(_slice)) - _ui_samples - (crossing-round(crossing)-offset)
            _time /= float(ui_samples)
            slices.append(_slice)
            times.append(_time)
    return times, slices

#
# For density plots
#

def float_to_raster_index(x_float, y_float, f_x0, f_y0, f_x1, f_y1, raster_height, raster_width):
    ''' (f_x0,f_y0), (f_x1,f_y1) are corners of rectangle defining space to convert to raster indexes
    where (f_x0,f_y0) is the lower left corner when plotted
    '''
    x_step = abs(f_x1 - f_x0)/(raster_width-1.0)
    y_step = abs(f_y1 - f_y0)/(raster_height-1.0)


    x_index = int(round((x_float - f_x0)/x_step))
    y_index = int(round((y_float - f_y0)/y_step))

    if x_index not in range(0, raster_width) or y_index not in range(0, raster_height):
        return None, None
    else:
        return x_index, y_index


def plot_raster_line(x0, y0, x1, y1, raster):
    """ Bresenham's algorithm for line rasterization
    """
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None

    n_rows, n_cols = raster.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 0
    if x0 < x1:
        sign_x = 1
    else:
        sign_x = -1
    sy = 0
    if y0 < y1:
        sign_y = 1
    else:
        sign_y = -1

    raster_error = dx - dy

    while True:
        # Note: this test is moved before setting
        # the value, so we don't set the last point.
        if x0 == x1 and y0 == y1:
            break

        if 0 <= x0 < n_rows and 0 <= y0 < n_cols:
            raster[x0, y0] += 1

        e2 = 2 * raster_error
        if e2 > -dy:
            raster_error -= dy
            x0 += sign_x
        if e2 < dx:
            raster_error += dx
            y0 += sign_y
