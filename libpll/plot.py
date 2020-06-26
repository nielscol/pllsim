""" Plotting methods for PLL analysis
    Cole Nielsen 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from libpll._signal import freq_to_index, make_signal
from libpll.analysis import fit_rw_pn, pll_tf_step, meas_inst_freq, ar_model
from libpll.filter import solpf, lf, pll_tf, pll_tf_, pi_pll_tf
from libpll.pncalc import min_ro_pn, tdc_pn, div_pn, lf_pn, bbpd_pn
from copy import copy


###############################################################################
# Plot style
###############################################################################

BOX_LINEWIDTH = 2
GRID_LINEWIDTH = 1
TICK_WIDTH = 2
GRAPH_LINEWIDTH = 2

def razavify(legend=True, legend_cols=2, labelsize=12, titlesize=12, loc="upper left",
            bbox_to_anchor=[0, 1]):
    """ Change plot style to look like something from Razavi's group
    """
    # bold everything
    font = {'weight' : 'bold',}
    plt.rc('font', **font)
    # make ticks inwards and thick
    ax = plt.gca()
    xax = ax.get_xaxis()
    yax = ax.get_yaxis()
    xax.set_ticks_position("both")
    yax.set_ticks_position("both")
    ax.tick_params(axis="both",which="both", direction="in", width=TICK_WIDTH)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(BOX_LINEWIDTH)
        plt.grid(linewidth=GRID_LINEWIDTH, color="dimgray", linestyle=":")
    # fix title/labels
    plt.title(ax.get_title(), fontweight="bold", size=titlesize)
    plt.xlabel(ax.get_xlabel(), fontweight="bold", size=labelsize)
    plt.ylabel(ax.get_ylabel(), fontweight="bold", size=labelsize)
    fontProperties = {'family':'sans-serif',
        'weight' : 'bold'}
    ax.set_xticklabels(ax.get_xticks(), fontProperties)
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    if legend:
        plt.legend(loc=loc, bbox_to_anchor=bbox_to_anchor,
                   ncol=legend_cols, shadow=False, title=None, fancybox=False,
                   frameon=True)
    for ln in ax.lines:
        ln.set_linewidth(GRAPH_LINEWIDTH)
    for text in ax.texts:
        text.set_fontweight("bold")

def adjust_subplot_space(rows, cols, wspace=0.15, hspace=0.15):
    """ Scales multiple subplot figure to aleviate text-overlapping
    """
    fig = plt.gcf()

    wspace = (wspace * cols) / (1 - wspace * cols - wspace)
    hspace = (hspace * rows) / (1 - hspace * rows - hspace)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)


###############################################################################
# AR model fit phase noise plotting
###############################################################################

def plot_pn_ar_model(signal, fmin=10, p=100, points=1001, tmin=None,
                     label="", tmax=None, verbose=True):
    if tmin or tmax:
        tmin = tmin if tmin else 0
        tmax = tmax if tmax else len(signal.td)/signal.fs
        nmin = int(round(tmin*signal.fs))
        nmax = int(round(tmax*signal.fs))
        td = signal.td[nmin:nmax]
    else: td = signal.td
    b, a = ar_model(td, p, fs=signal.fs)
    w_min = 2*fmin/signal.fs
    w, h = scipy.signal.freqz(b, a, np.geomspace(np.pi*w_min, np.pi, points))
    f = np.geomspace(fmin, signal.fs/2, points)
    plt.semilogx(f, 20*np.log10(np.abs(h)), label="AR(%d) Fit %s"%(p, label))
    plt.title("SSB Phase Noise [dBc/Hz]")
    plt.xlabel("Frequency Offset [Hz]")
    plt.ylabel("Relative Power [dBc/Hz]")

###############################################################################
# Phase noise plotting
###############################################################################

def plot_pn_ssb(signal, f0, fref=None, dfmin=None, dfmax=None,
                tmin=None, tmax=None, line_fit=True, verbose=True):
    """ plot single-side band (SSB) phase noise
        args:
            signal - Signal class object with oscillator time domain signal
            f0 - fundemental tone of oscillator
            line_fit - fit 1/f line to phase noise
    """
    if tmin or tmax:
        tmin = tmin if tmin else 0
        tmax = tmax if tmax else len(signal.td)/signal.fs
        nmin = int(round(tmin*signal.fs))
        nmax = int(round(tmax*signal.fs))
        td = signal.td[nmin:nmax]
    else:
        td = signal.td
    fbin = signal.fs/len(td)
    # calculate SSB spectrum
    carrier_index = freq_to_index(signal, abs(f0))
    bin_offset = f0 - carrier_index*fbin
    window = scipy.signal.windows.blackmanharris(len(td))
    BH_LOBE_W = 4
    fd = np.fft.fft(window*td)
    carrier = np.abs(fd[carrier_index])
    ssb_slice = copy(fd[carrier_index+1:int(round(len(td)/2))])
    ssb_slice /= carrier
    ssb_slice_f = (np.arange(len(ssb_slice))+1)*fbin - bin_offset
    if dfmax:
        _ssb_slice_f = ssb_slice_f[ssb_slice_f <= dfmax]
        _ssb_slice  = ssb_slice[:len(_ssb_slice_f)]
    else:
        _ssb_slice_f = ssb_slice_f
        _ssb_slice = ssb_slice
    # plot SSB spectrum
    plt.semilogx(_ssb_slice_f, 20*np.log10(np.abs(_ssb_slice)))
    # fit 1/f phase noise model to PN
    if line_fit:
        if fref:
            n = 1
            while True:
                spur_index = int(round((n*fref+bin_offset)/fbin))-1
                if spur_index+BH_LOBE_W >= len(ssb_slice_f):
                    break
                ssb_slice[spur_index-BH_LOBE_W:spur_index+BH_LOBE_W+1] = np.zeros(2*BH_LOBE_W+1)
                n += 1
        pn_model = fit_rw_pn(ssb_slice_f, ssb_slice)
        ssb_model = pn_model(_ssb_slice_f)
        plt.semilogx(_ssb_slice_f, 20*np.log10(ssb_model))
    plt.grid()
    plt.title("SSB Phase Noise [dBc/Hz]")
    plt.xlabel("Frequency Offset [Hz]")
    plt.ylabel("Relative Power [dBc/Hz]")

def plot_pn_ssb2(signal, fref=None, dfmin=None, dfmax=None,
                 tmin=None, tmax=None, line_fit=False,
                 label="", verbose=True):
    """ Plot phase noise spectrum for time domain PHASE NOISE signal
    """
    # window = scipy.signal.windows.blackmanharris(len(signal.td))
    # fd = np.fft.fft(window*signal.td)
    if tmin or tmax:
        tmin = tmin if tmin else 0
        tmax = tmax if tmax else len(signal.td)/signal.fs
        nmin = int(round(tmin*signal.fs))
        nmax = int(round(tmax*signal.fs))
        fd = np.fft.fft(signal.td[nmin:nmax])
    elif not any(signal.fd): fd = np.fft.fft(signal.td)
    else: fd = signal.fd
    fbin = signal.fs/len(fd)
    ssb_slice = copy(fd[1:int(round(len(fd)/2))])
    # ssb_slice /= float(len(fd))
    ssb_slice /= np.sqrt(len(fd)*signal.fs) # normalize for PSD
    ssb_slice_f = (np.arange(len(ssb_slice))+1)*fbin
    if dfmax:
        _ssb_slice_f = ssb_slice_f[ssb_slice_f <= dfmax]
        _ssb_slice  = ssb_slice[:len(_ssb_slice_f)]
    else:
        _ssb_slice_f = ssb_slice_f
        _ssb_slice = ssb_slice
    if dfmin:
       subset = np.where(_ssb_slice_f>dfmin)
       _ssb_slice_f =_ssb_slice_f[subset]
       _ssb_slice = _ssb_slice[subset]
    # plot SSB spectrum
    plt.semilogx(_ssb_slice_f, 20*np.log10(np.abs(_ssb_slice)), label="DFT %s"%label)
    # fit 1/f phase noise model to PN
    if line_fit:
        if fref:
            n = 1
            while True:
                spur_index = int(round((n*fref)/fbin))-1
                if spur_index >= len(ssb_slice_f):
                    break
                ssb_slice[spur_index] = 0
                n += 1
        pn_model = fit_rw_pn(ssb_slice_f, ssb_slice)
        ssb_model = pn_model(_ssb_slice_f)
        plt.semilogx(_ssb_slice_f, 20*np.log10(ssb_model), label="Line fit")
    # M_BARTLETT = len(signal.td) - 128
    # n_avg = len(signal.td) - M_BARTLETT -1
    # avg = np.zeros(M_BARTLETT, dtype=complex)
    # for n in range(n_avg):
    #     avg += np.fft.fft(signal.td[n:n+M_BARTLETT]/float(M_BARTLETT))**2
    # avg /= float(n_avg)
    # fbin = signal.fs/float(M_BARTLETT)
    # freqs = np.arange(1, int(M_BARTLETT/2))*fbin
    # plt.semilogx(freqs, 10*np.log10(avg[1:int(M_BARTLETT/2)]))
    plt.grid()
    plt.title("SSB Phase Noise [dBc/Hz]")
    plt.xlabel("Frequency Offset [Hz]")
    plt.ylabel("Relative Power [dBc/Hz]")


###############################################################################
# Time domain plotting
###############################################################################

def plot_td(signal, verbose=True, tmin=None, tmax=None, label="", title="",
            dots=False, alpha=1.0, *args, **kwargs):
    """ Plots time domain data for signal
    """
    if verbose:
        print("\n* Plotting signal in time domain")
        print("\tSignal.name = %s"%signal.name)
    times = np.arange(signal.samples)/float(signal.fs)
    plt.ylabel("Signal")
    plt.xlabel("Time [s]")
    plt.grid()
    nmin = int(round(tmin*signal.fs)) if tmin else 0
    nmax = int(round(tmax*signal.fs)) if tmax else -1
    tmin = tmin if tmin else 0
    tmax = tmax if tmax else times[-1]
    if label != "":
        plt.plot(times[nmin:nmax], signal.td[nmin:nmax], label=label, alpha=alpha)
        plt.legend()
    else:
        plt.plot(times[nmin:nmax], signal.td[nmin:nmax], label=signal.name, alpha=alpha)
    if dots:
        plt.scatter(times[nmin:nmax], signal.td[nmin:nmax], alpha=alpha)
    plt.xlim((tmin,tmax))
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


###############################################################################
# Frequency domain plotting
###############################################################################

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
    dbsa = 20*np.log10(signal.samples)
    if log:
        if verbose:
            print("\tFFT - Log [dB] scale")
        plt.ylabel("FFT(Signal) [dB]")
        plt.plot(freqs, 20*np.log10(np.abs(np.fft.fftshift(signal.fd)))-dbsa, label=label, alpha=alpha)
    else:
        if verbose:
            print("\tFFT - Magnitude")
        plt.ylabel("FFT(Signal) [magnitude]")
        plt.plot(freqs, np.abs(np.fft.fftshift(signal.fd)), label = label, alpha=alpha)
    if label != "":
        plt.legend()
    plt.title("Power Spectral Density "+title)
    plt.legend()

###############################################################################
# Loop filter transfer function plotting
###############################################################################

def plot_osc_pn_ideal(pn_dco, pn_dco_df, steps=100, fmin=1, fmax=1e7, *args, **kwargs):
    freqs = np.geomspace(fmin, fmax, int(steps))
    s0_osc = pn_dco*pn_dco_df**2
    psd = s0_osc/freqs**2
    plt.semilogx(freqs, 10*np.log10(psd), label="Nom. Osc Noise")


def plot_pi_pll_bbpd_pn(k, fz, delay, int_pn, fref, steps=100, fmin=1, fmax=1e7, *args, **kwargs):
    freqs = np.geomspace(fmin, fmax, int(steps))
    g = pi_pll_tf(freqs, k, fz, delay)
    psd = int_pn*(np.pi/2 - 1)*np.abs(g)**2/fref
    plt.semilogx(freqs, 10*np.log10(psd), label="Est. BBPD Noise")

def plot_pi_pll_osc_pn(pn_dco, pn_dco_df, k, fz, delay, int_pn, fref, steps=100, fmin=1, fmax=1e7,
                       *args, **kwargs):
    freqs = np.geomspace(fmin, fmax, int(steps))
    g = pi_pll_tf(freqs, k, fz, delay)
    s0_osc = pn_dco*pn_dco_df**2
    psd = s0_osc*np.abs(1-g)**2/freqs**2
    plt.semilogx(freqs, 10*np.log10(psd), label="Est. Osc Noise")


def plot_loop_filter(pn_dco, pn_dco_df, fclk, _type, m, n, k, ki, fz, fp, delay, bw,
                     a0, a1, b0, b1, b2, mode="tdc", sigma_ph=0.1, steps=100, fmax=1e7,
                     *args, **kwargs):
    freqs = np.geomspace(1, fmax, int(steps))
    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)
    kpn = pn_dco*pn_dco_df**2

    plt.subplot(1,3,1)
    # plt.semilogx(freqs, 20*np.log10(np.abs(solpf(freqs, fn, damping))), label="Ideal")
    plt.semilogx(freqs, 20*np.log10(np.abs(g)), label="G(f)")
    plt.semilogx(freqs, 20*np.log10(np.abs(1-g)), label="1-G(f)")
    plt.legend()
    plt.grid()
    plt.title("Closed loop responses")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gain [dB]")

    plt.subplot(1,3,2)
    # ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))
    if mode is "tdc": ntdc = tdc_pn(fclk, n, m, g)
    if mode is "bbpd": ntdc = bbpd_pn(fclk, n, sigma_ph, g)

    plt.semilogx(freqs, 10*np.log10(ntdc), label="TDC")
    plt.semilogx(freqs, 10*np.log10(ndco), label="DCO")
    plt.semilogx(freqs, 10*np.log10(ndco+ntdc), label="Combined")
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")

    plt.subplot(1,3,3)
    w, h = scipy.signal.freqz([a0, a1], [b0, b1, b2], fs=fclk)
    _h = lf(w[1:], _type, ki, fz, fp)
    plt.title("Ideal versus discrete loop filter")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gain [dB]")
    plt.semilogx(w[1:], 20*np.log10(np.abs(h[1:])), label="Discrete")
    plt.semilogx(w[1:], 20*np.log10(np.abs(_h)), label="Ideal")
    plt.grid()
    plt.legend()

def plot_dco_pn(pn_dco, pn_dco_df, fclk, _type, m, n, k, ki, fz, fp, delay, bw,
                a0, a1, b0, b1, b2, steps=100, fmax=1e7, label="",
                *args, **kwargs):
    freqs = np.geomspace(1, fmax, int(steps))

    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)
    kpn = pn_dco*pn_dco_df**2

    # ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0

    plt.semilogx(freqs, 10*np.log10(ndco), label="DCO %s"%label)
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")

def plot_tdc_pn(pn_dco, pn_dco_df, fclk, _type, m, n, k, ki, fz, fp, delay, bw,
                a0, a1, b0, b1, b2, steps=100, fmax=1e7, label="",
                *args, **kwargs):
    freqs = np.geomspace(1, fmax, int(steps))

    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)

    ntdc = tdc_pn(fclk, n, m, g)
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))

    plt.semilogx(freqs, 10*np.log10(ntdc), label="TDC %s"%label)
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")

def plot_total_pn(pn_dco, pn_dco_df, fclk, _type, m, n, k, ki, fz, fp, delay, bw,
                a0, a1, b0, b1, b2, steps=100, fmax=1e7, label="",
                *args, **kwargs):
    freqs = np.geomspace(1, fmax, int(steps))

    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)
    kpn = pn_dco*pn_dco_df**2

    # ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    ntdc = tdc_pn(fclk, n, m, g)
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))

    plt.semilogx(freqs, 10*np.log10(ndco+ntdc), label="Total %s"%label)
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")


def plot_lf_ideal_pn(pn_dco, pn_dco_df, fclk, _type, m, n, k, fz, fp, delay,
                     mode="tdc", sigma_ph=0.1, steps=100, fmin=1e1, fmax=1e7,
                     *args, **kwargs):
    df = fmax/steps
    freqs = np.geomspace(fmin, fmax, int(steps))

    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf_(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)
    kpn = pn_dco*pn_dco_df**2

    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    # ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    if mode is "tdc": ntdc = tdc_pn(fclk, n, m, g)
    if mode is "bbpd": ntdc = bbpd_pn(fclk, n, sigma_ph, g)
    # ntdc = tdc_pn(fclk, n, m, g)
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))

    plt.semilogx(freqs, 10*np.log10(ntdc), label="Detector - ideal")
    plt.semilogx(freqs, 10*np.log10(ndco), label="DCO - ideal")
    plt.semilogx(freqs, 10*np.log10(ndco+ntdc), label="Total - ideal",color="C6")
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")



def plot_lf_ideal_pn_full(div_jit, pn_dco, pn_dco_df, kdco, fclk, _type, m, n, k, fz, fp, delay,
                          b1, b2, steps=100, fmin=1e1, fmax=1e7, *args, **kwargs):
    df = fmax/steps
    freqs = np.geomspace(fmin, fmax, int(steps))
    lf_params = dict(_type=_type, k=k, fz=fz, fp=fp, delay=delay)
    # a = pll_otf2(freqs, M, N, KDCO, KP, FZ, FP, DELAY)
    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)
    kpn = pn_dco*pn_dco_df**2

    ndco = np.abs(1-g)**2*kpn/freqs**2
    ndco[np.where(g==1)] = 0
    # ndco = min_ro_pn(fclk*n, freqs, pn_dco, pn_dco_df)*np.abs(1-g)**2
    ntdc = tdc_pn(fclk, n, m, g)
    # ntdc = tdc_pn(fclk, n, g, 1/float(m*fclk))

    nlf = lf_pn(freqs, fclk, lf_params, kdco, b1, b2)
    ndiv = div_pn(fclk, n, div_jit, g)


    plt.semilogx(freqs, 10*np.log10(ntdc), label="TDC")
    plt.semilogx(freqs, 10*np.log10(ndco), label="DCO")
    plt.semilogx(freqs, 10*np.log10(nlf), label="Loop filter")
    plt.semilogx(freqs, 10*np.log10(ndiv), label="Divider")
    plt.semilogx(freqs, 10*np.log10(ndco+ntdc+nlf+ndiv), label="Total")
    plt.legend()
    plt.grid()
    plt.title("SSB Phase noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase noise [dBc]")


def plot_pll_tf(k, fz, fp, _type, delay, steps=100, fmax=1e7, *args, **kwargs):
    freqs = np.geomspace(1, fmax, int(steps))

    g = pll_tf(freqs, _type, k, fz, fp, delay)
    # g = a/(1+a)

    plt.subplot(1,3,1)
    plt.semilogx(freqs, 20*np.log10(np.abs(g)), label="G(f)")
    plt.semilogx(freqs, 20*np.log10(np.abs(1-g)), label="1-G(f)")
    plt.legend()
    plt.grid()
    plt.title("Closed loop responses")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gain [dB]")

###############################################################################
# Step response plotting
###############################################################################

def plot_pll_tf_step(pll_tf, tmax):
    step = pll_tf_step(pll_tf, tmax)
    plot_td(step)
    plt.title("Step response")

###############################################################################
# Time domain frequency measurement
###############################################################################

def plot_pllsim_lf_inst_freq(pllsim_data):
    inst_freq = make_signal(td=pllsim_data["lf"].td*pllsim_data["params"]["kdco"], fs = pllsim_data["lf"].fs)
    plot_td(inst_freq)
    plt.ylabel("Frequency [Hz]")

def plot_pllsim_osc_inst_freq(pllsim_data):
    plot_td(meas_inst_freq(pllsim_data["osc"]))
    plt.ylabel("Frequency [Hz]")
