import numpy as np
import matplotlib.pyplot as plt
from libradio._signal import wav_to_signal, save_signal_to_wav, make_signal
from libradio.sync import frame_data_bursts, get_precomputed_codes, detect_sync, make_sync_fir
from libradio.transforms import convert_to_bitstream, ulaw_compress, ulaw_expand
from libradio.transforms import butter_lpf, cheby2_lpf, fir_filter, fir_correlate, add_noise, filter_and_downsample, gaussian_fade, fft_resample
from libradio.modulation import generate_gmsk_baseband, upconvert_baseband, demodulate_gmsk
from libradio.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter, kaiser_composite_tx_rx_filter, rcos_composite_tx_rx_filter, gmsk_tx_filter
from libradio.plot import plot_td, plot_fd
from libpll.analysis import pn_signal

def meas_ber(data, pattern):
    bits = 0
    errors = 0
    for frame in data:
        len_pat = len(pattern)
        n_pat = int(len(frame)/len(pattern))
        for n in range(n_pat):
            for m in range(len_pat):
                bits += 1
                if pattern[m] != frame[n*len_pat+m]:
                    errors += 1
    return errors/float(bits)

_prbs7 = [1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0]
prbs7 = []
for b in _prbs7:
    if b:
        prbs7.extend([1,1,1,1])
    else:
        prbs7.extend([0,0,0,0])


BITRATE = 1000000          # Audio rate + sync overhead
SPECTRAL_EFF = 1.0              # bits/s/Hz, 1.0 for MSK and GMSK
BW = BITRATE/(SPECTRAL_EFF)     # theoretical signal bandwidth
RX_LPF = 1.0*BW                 # Corner freqency of LPF on Rx I/Q to reject aliasing/interferers
RX_LPF_TYPE = "butter"          # Filter type "butter" or "cheby2"
ORDER = 5

iq_file = "prbs7_gmsk_iq_bt_0_3_oversamp_8.csv"

DELAY = 0.15 # second
DURATION = 1.0 # second
FILE = "./sample_audio/all.wav"
SYNC_CODE_LEN = 24
FRAME_PAYLOAD = 640*4
AUDIO_RATE = 1000000
BITRATE = 1000000
TX_RATE = 1000000
SYNC_POS = "center"
OVERSAMPLING = 8
IQ_RATE = BITRATE*OVERSAMPLING
BT_TX = 0.3
BT_COMPOSITE = 1.0              # response of combined Rx+Tx filter
TX_FIR_SPAN = 8
RX_FIR_SPAN = 8.0               # Extent of Rx matched FIR filter in # of symbols
SYNC_PULSE_SPAN = 8             # FIR span in symbols of pulse shape used to filter sync code for 
BLOCKS_PER_S = 10
ATTEMPTS = 100
# Read source audio file and scale to fit range of sample size
# audio = wav_to_signal(FILE)
# samples = int(audio.fs*DURATION)
# delay = int(audio.fs*DELAY)
# audio.td = audio.td[delay:delay+samples]

# ramp = np.arange(2**13)*2**3 - 2**15
# ramp = ramp.astype(np.int16)
# ramp = make_signal(td=ramp, fs=AUDIO_RATE, signed=True, bits=16)
# plt.plot(ramp.td)
# plt.show()

# block_audio_sa = int(AUDIO_RATE/(BLOCKS_PER_S*OVERSAMPLING))
# num_blocks = int(len(ramp.td)/block_audio_sa)

FRAME_PER_SEC = 100
FRAMES = 100
# TX_LEN = OVERSAMPLING*4*2044

data = np.zeros(FRAME_PAYLOAD*FRAMES)
for n in range(FRAMES):
    for m in range(int(FRAME_PAYLOAD/len(prbs7))):
        data[n*FRAME_PAYLOAD+m*len(prbs7):n*FRAME_PAYLOAD+(m+1)*len(prbs7)] = prbs7
data = [1.0 if x else -1.0 for x in data]
bitstream = make_signal(td=data)

# coded = ulaw_compress(ramp)
# recovered = ulaw_expand(coded)
# save_signal_to_wav(recovered, "ramp.wav")
# bitstream = convert_to_bitstream(coded)


# for n in range(num_blocks):
#     print(n, num_blocks)
#     plt.subplot(2,5,n+1)
#     plt.plot(recovered.td[n*block_audio_sa:(n+1)*block_audio_sa])
#     plt.title("Waveform %d"%(n+1))

# plt.show()

sync_codes = get_precomputed_codes()
sync_code = sync_codes[SYNC_CODE_LEN]
message = frame_data_bursts(bitstream, sync_code, FRAME_PAYLOAD, TX_RATE, BITRATE, FRAME_PER_SEC, sync_pos=SYNC_POS)


kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                           fs=IQ_RATE, norm=True)
fir_matched_kaiser = gmsk_matched_kaiser_rx_filter(OVERSAMPLING, RX_FIR_SPAN, BT_TX,
                                                   BT_COMPOSITE, fs=IQ_RATE)
sync_fir_kaiser = make_sync_fir(sync_code, kaiser_fir, OVERSAMPLING)


cnrs = np.linspace(0,12,21)
bers = []
for cnr in cnrs:
    _bers = []
    for attempt in range(ATTEMPTS):
        gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                                pulse_span=TX_FIR_SPAN, keep_extra=True, binary_message=False)

    # demod = demodulate_gmsk(gmsk_i, gmsk_q, OVERSAMPLING)
    #plot_td(demod)
    #np.savetxt("demod_burst.csv", demod.td, delimiter=",")

    # i = np.ones(TX_LEN)
    # q = np.zeros(TX_LEN)
    # i = gmsk_i.td
    # q = gmsk_q.td
    # i[:len(gmsk_i.td)] = gmsk_i.td
    # q[:len(gmsk_q.td)] = gmsk_q.td

    # plt.plot(i)
    # plt.plot(q)
    # plt.show()
        gmsk_i.td += np.random.normal(0,10**(-cnr/20), len(gmsk_i.td))
        gmsk_q.td += np.random.normal(0,10**(-cnr/20), len(gmsk_q.td))

        # gmsk_i = fft_resample(gmsk_i, n_samples=len(gmsk_i.td)*2) # force rf and interferer to same len
        # gmsk_q = fft_resample(gmsk_q, n_samples=len(gmsk_q.td)*2) # force rf and interferer to same len

        # gmsk_i = filter_and_downsample(gmsk_i, n=2)
        # gmsk_q = filter_and_downsample(gmsk_q, n=2)
        gmsk = np.asarray([gmsk_i.td, gmsk_q.td])
        # np.savetxt(iq_file, gmsk.T, delimiter=",")



        if RX_LPF:
            if RX_LPF_TYPE == "butter":
                rx_i = butter_lpf(gmsk_i, cutoff = RX_LPF, order=ORDER)
                rx_q = butter_lpf(gmsk_q, cutoff = RX_LPF, order=ORDER)
            elif RX_LPF_TYPE == "cheby2":
                rx_i = cheby2_lpf(gmsk_i, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)
                rx_q = cheby2_lpf(gmsk_q, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)

        demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)
        #plt.figure(1)
        #plot_td(demod)

        demod_kaiser = fir_filter(demodulated, fir_matched_kaiser, OVERSAMPLING)
        #TX_LO = TX_RATE*3.0
        #rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q)
        #plt.figure(2)
        #plot_fd(rf)
        # plt.plot(demod_kaiser.td)


        sync_correl_kaiser = fir_correlate(demod_kaiser, sync_fir_kaiser, OVERSAMPLING)
        peak_i, peak_v =  detect_sync(sync_correl_kaiser, sync_code, FRAME_PAYLOAD, OVERSAMPLING)
        # print(peak_i, peak_v)
        # plt.plot(sync_correl_kaiser.td)

        OFFSET=int(RX_FIR_SPAN/2)
        data = []
        x = 1
        for i,v in zip(peak_i, peak_v):
            if x:
                # print("***", i)
                curr = np.zeros(FRAME_PAYLOAD)
                avgd = np.zeros(int(FRAME_PAYLOAD/4))
                half_frame = int(FRAME_PAYLOAD/2)
                offsets_a = (np.arange(half_frame)-half_frame+OFFSET)*OVERSAMPLING+i
                offsets_a = offsets_a.astype(int)
                offsets_b = (np.arange(half_frame)+SYNC_CODE_LEN+OFFSET)*OVERSAMPLING+i
                offsets_b = offsets_b.astype(int)
                # print(offsets_a, offsets_b)
                if not min(offsets_a) < 0 and not max(offsets_b) > len(demod_kaiser.td):
                    curr[0:half_frame] = demod_kaiser.td[offsets_a]
                    curr[half_frame:] = demod_kaiser.td[offsets_b]
                    # print(curr)
                    for n in range(int(FRAME_PAYLOAD/4)):
                        avgd[n] = np.mean(curr[4*n:4*n+4])
                    data.append(avgd)
            x = 1

        data = [[1 if b>=0.0 else 0 for b in frame] for frame in data]

        # for frame in data:
            # print(frame[:10])

        curr_ber = meas_ber(data, _prbs7)
        # bers.append(curr_ber)
        _bers.append(curr_ber)
        print("CNR = %.2f dB,\tN = %d\tBER = %E"%(cnr, attempt, curr_ber))
    bers.append(np.average(_bers))

plt.semilogy(cnrs, bers)
plt.title("GFSK BER, BT=0.3, 1 MSymbol/s, $\pm$ 500 Khz deviation\n4 symbol averaging (250 kbps equivalent)")
plt.xlabel("CNR (dB)")
plt.ylabel("BER")
plt.grid()
plt.show()

import json
with open("gmsk_ber_data.json", "w") as f:
    json.dump(dict(cnr=list(cnrs), ber=list(bers)), f)


foo()

SAVE_FILE = "lin_pll_simulation.pickle"

import pickle
pickle_in = open(SAVE_FILE,"rb")
sim_data = pickle.load(pickle_in)
lf_params = sim_data["lf_params"]
lf_params_bbpd = sim_data["lf_params_bbpd"]
sigma_ph = sim_data["sigma_ph"]
DCO_PN = sim_data["dco_pn"]
main_pn_data = sim_data["sim_data"]
DIV_N = 150
pn_sig = pn_signal(main_pn_data, DIV_N)



gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                        pulse_span=TX_FIR_SPAN, keep_extra=True, binary_message=False)

# demod = demodulate_gmsk(gmsk_i, gmsk_q, OVERSAMPLING)
#plot_td(demod)
#np.savetxt("demod_burst.csv", demod.td, delimiter=",")

# i = np.ones(TX_LEN)
# q = np.zeros(TX_LEN)
# i = gmsk_i.td
# q = gmsk_q.td
# i[:len(gmsk_i.td)] = gmsk_i.td
# q[:len(gmsk_q.td)] = gmsk_q.td

# plt.plot(i)
# plt.plot(q)
# plt.show()
# gmsk_i.td += np.random.normal(0,10**(-cnr/20), len(gmsk_i.td))
# gmsk_q.td += np.random.normal(0,10**(-cnr/20), len(gmsk_q.td))


gmsk_i = fft_resample(gmsk_i, n_samples=len(gmsk_i.td)*2) # force rf and interferer to same len
gmsk_q = fft_resample(gmsk_q, n_samples=len(gmsk_q.td)*2) # force rf and interferer to same len
lo = np.cos(pn_sig.td[-len(gmsk_i.td):])
plt.plot(pn_sig.td[-len(gmsk_i.td):])
plt.plot(lo)
plt.show()
print("***", len(gmsk_i.td), len(lo), len(pn_sig.td))
# lo_q = np.cos(pn_sig.td[-len(gmsk_i.td):])
gmsk_i.td *= lo
gmsk_q.td *= lo
lo = np.sin(pn_sig.td[-len(gmsk_i.td):])
cnr = -20*np.log10(np.std(lo))
# foo()
gmsk_i = filter_and_downsample(gmsk_i, n=2)
gmsk_q = filter_and_downsample(gmsk_q, n=2)
gmsk = np.asarray([gmsk_i.td, gmsk_q.td])
# np.savetxt(iq_file, gmsk.T, delimiter=",")



if RX_LPF:
    if RX_LPF_TYPE == "butter":
        rx_i = butter_lpf(gmsk_i, cutoff = RX_LPF, order=ORDER)
        rx_q = butter_lpf(gmsk_q, cutoff = RX_LPF, order=ORDER)
    elif RX_LPF_TYPE == "cheby2":
        rx_i = cheby2_lpf(gmsk_i, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)
        rx_q = cheby2_lpf(gmsk_q, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)

demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)
#plt.figure(1)
#plot_td(demod)

demod_kaiser = fir_filter(demodulated, fir_matched_kaiser, OVERSAMPLING)
#TX_LO = TX_RATE*3.0
#rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q)
#plt.figure(2)
#plot_fd(rf)
# plt.plot(demod_kaiser.td)


sync_correl_kaiser = fir_correlate(demod_kaiser, sync_fir_kaiser, OVERSAMPLING)
peak_i, peak_v =  detect_sync(sync_correl_kaiser, sync_code, FRAME_PAYLOAD, OVERSAMPLING)
# print(peak_i, peak_v)
# plt.plot(sync_correl_kaiser.td)

OFFSET=int(RX_FIR_SPAN/2)
data = []
x = 1
for i,v in zip(peak_i, peak_v):
    if x:
        # print("***", i)
        curr = np.zeros(FRAME_PAYLOAD)
        avgd = np.zeros(int(FRAME_PAYLOAD/4))
        half_frame = int(FRAME_PAYLOAD/2)
        offsets_a = (np.arange(half_frame)-half_frame+OFFSET)*OVERSAMPLING+i
        offsets_a = offsets_a.astype(int)
        offsets_b = (np.arange(half_frame)+SYNC_CODE_LEN+OFFSET)*OVERSAMPLING+i
        offsets_b = offsets_b.astype(int)
        # print(offsets_a, offsets_b)
        if not min(offsets_a) < 0 and not max(offsets_b) > len(demod_kaiser.td):
            curr[0:half_frame] = demod_kaiser.td[offsets_a]
            curr[half_frame:] = demod_kaiser.td[offsets_b]
            # print(curr)
            for n in range(int(FRAME_PAYLOAD/4)):
                avgd[n] = np.mean(curr[4*n:4*n+4])
            data.append(avgd)
    x = 1

data = [[1 if b>=0.0 else 0 for b in frame] for frame in data]

# for frame in data:
    # print(frame[:10])

curr_ber = meas_ber(data, _prbs7)
bers.append(curr_ber)
print("PLL CNR = %d dB, BER = %E"%(cnr, curr_ber))

