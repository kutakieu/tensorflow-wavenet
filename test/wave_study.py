import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

NOTES = ['D#3', 'G3', 'A#3']  # e-flat chord
NOTES_HZ = librosa.note_to_hz(NOTES)

SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 400
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
GENERATE_SAMPLES = 900
QUANTIZATION_CHANNELS = 256
NUM_SPEAKERS = 3
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz
#
# def plot_waveforms(waveform, sfft=False, title=''):
#     waveform = np.squeeze(waveform)
#     if waveform.ndim < 2:
#         waveform = np.reshape(waveform, (1, -1))
#         legend = [' + '.join(NOTES)]
#     else:
#         legend = [("{} {:.1f} Hz".format(note, hz)) for (note, hz) in zip(NOTES, NOTES_HZ)]
#
#     if sfft:
#         plt.figure(title + ' log STFT Magnitude')
#     else:
#         plt.figure(title + ' Power Spectrum')
#
#     no_waves = np.shape(waveform)[0]
#     for i in range(no_waves):
#         if sfft:
#             npersg = round(0.2*SAMPLE_RATE_HZ)
#             f, t, Zxx = signal.stft(waveform[i], SAMPLE_RATE_HZ, nperseg=npersg, window='blackman')
#             plt.subplot(1, no_waves, i+1)
#             plt.pcolormesh(t, f, np.log(np.abs(Zxx)))
#             plt.xticks([])
#             plt.xlabel(legend[i])
#         else:
#             power_spectrum = np.abs(np.fft.fft(waveform[i]))
#             freqs = np.fft.fftfreq(np.shape(waveform[i])[0], 1.0 / SAMPLE_RATE_HZ)
#             indices = np.argsort(freqs)
#             margin = 50
#             indices = [i for i in indices if (min(NOTES_HZ)-margin <= freqs[i] <= max(NOTES_HZ)+margin)]
#             plt.subplot(2, 1, 1)
#             plt.ylabel('Power')
#             plt.xlabel('Frequency [Hz]')
#             plt.autoscale(enable=True, axis='both', tight=True)
#             plt.plot(freqs[indices], power_spectrum[indices])
#             plt.subplot(2, 1, 2)
#             plt.ylabel('Amplitude')
#             plt.xlabel('Sample')
#             plt.plot(waveform[i])
#     plt.legend(legend, loc='best', fancybox=True)
#     plt.show(block=False)


sample_period = 1.0/SAMPLE_RATE_HZ
times = np.arange(0.0, SAMPLE_DURATION, sample_period)
E_f = 0.6 * np.sin(times * 2.0 * np.pi * F1)
G = 0.6 * np.sin(times * 2.0 * np.pi * F2)
B_f = 0.6 * np.sin(times * 2.0 * np.pi * F3)

full_wave = np.concatenate((E_f, G, B_f))

no_waves = np.shape(full_wave)[0]

npersg = round(0.2*SAMPLE_RATE_HZ)
f, t, Zxx = signal.stft(full_wave, SAMPLE_RATE_HZ, nperseg=npersg, window='blackman')
plt.subplot(1, no_waves, i+1)
plt.pcolormesh(t, f, np.log(np.abs(Zxx)))
plt.xticks([])
# plt.xlabel(legend[i])

power_spectrum = np.abs(np.fft.fft(full_wave))
freqs = np.fft.fftfreq(np.shape(full_wave)[0], 1.0 / SAMPLE_RATE_HZ)
indices = np.argsort(freqs)
margin = 50
indices = [i for i in indices if (min(NOTES_HZ)-margin <= freqs[i] <= max(NOTES_HZ)+margin)]
plt.subplot(2, 1, 1)
plt.ylabel('Power')
plt.xlabel('Frequency [Hz]')
plt.autoscale(enable=True, axis='both', tight=True)
plt.plot(freqs[indices], power_spectrum[indices])
plt.subplot(2, 1, 2)
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.plot(full_wave)