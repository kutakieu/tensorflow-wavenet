"""Unit tests for the WaveNet that check that it can train on audio data."""
import json
import numpy as np
import sys
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import librosa
from scipy import signal

from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode, image2vector)

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
receptive_field = 256

def make_sine_waves(global_conditioning, local_conditioning=None, randomness=None, noisy=None, generate_two_waves=None, batch_size=3):
    """Creates a time-series of sinusoidal audio amplitudes."""
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    if global_conditioning and not local_conditioning:
        LEADING_SILENCE = random.randint(10, 128)
        amplitudes = np.zeros(shape=(NUM_SPEAKERS, len(times)))
        amplitudes[0, 0:LEADING_SILENCE] = 0.0
        amplitudes[1, 0:LEADING_SILENCE] = 0.0
        amplitudes[2, 0:LEADING_SILENCE] = 0.0
        start_time = LEADING_SILENCE / SAMPLE_RATE_HZ
        times = times[LEADING_SILENCE:] - start_time
        amplitudes[0, LEADING_SILENCE:] = 0.6 * np.sin(times *
                                                       2.0 * np.pi * F1)
        amplitudes[1, LEADING_SILENCE:] = 0.5 * np.sin(times *
                                                       2.0 * np.pi * F2)
        amplitudes[2, LEADING_SILENCE:] = 0.4 * np.sin(times *
                                                       2.0 * np.pi * F3)
        gc = np.zeros((NUM_SPEAKERS, 1), dtype=np.int)
        gc[0, 0] = 0
        gc[1, 0] = 1
        gc[2, 0] = 2
    elif global_conditioning and local_conditioning and not randomness:
        # lc_0 = np.full(len(times), 0)
        # lc_1 = np.full(len(times), 1)
        # lc_2 = np.full(len(times), 2)

        # TODO: experiment the case set lc=0 for the initial silences
        # set lc=0 for the initial silence
        lc_0 = np.full(len(times), 1)
        lc_1 = np.full(len(times), 2)
        lc_2 = np.full(len(times), 3)

        lc = np.stack((lc_0, lc_1, lc_2))

        LEADING_SILENCE = random.randint(10, 128)
        amplitudes = np.zeros(shape=(NUM_SPEAKERS, len(times)))
        amplitudes[0, 0:LEADING_SILENCE] = 0.0
        amplitudes[1, 0:LEADING_SILENCE] = 0.0
        amplitudes[2, 0:LEADING_SILENCE] = 0.0

        start_time = LEADING_SILENCE / SAMPLE_RATE_HZ
        times = times[LEADING_SILENCE:] - start_time

        amplitudes[0, LEADING_SILENCE:] = 0.6 * np.sin(times * 2.0 * np.pi * F1)
        amplitudes[1, LEADING_SILENCE:] = 0.5 * np.sin(times * 2.0 * np.pi * F2)
        amplitudes[2, LEADING_SILENCE:] = 0.4 * np.sin(times * 2.0 * np.pi * F3)

        gc = np.zeros((NUM_SPEAKERS, 1), dtype=np.int)
        gc[0, 0] = 0
        gc[1, 0] = 0
        gc[2, 0] = 0

    elif global_conditioning and local_conditioning and randomness:

        # amplitudes = np.zeros((batch_size, len(times)))
        # # lc = np.zeros((batch_size*sample_num, len(times)))
        # lc = np.zeros((batch_size, len(times)))
        # # gc = np.zeros(batch_size*sample_num)
        # gc = np.zeros(batch_size)
        #
        # # amplitudes[:,0] = audio
        # # lc[:,0] = current_lc
        # # lc = np.stack(current_lc)
        # # speaker_ids = np.stack([0])
        #
        # note1 = 0.6 * np.sin(times * 2.0 * np.pi * F1)
        # note2 = 0.5 * np.sin(times * 2.0 * np.pi * F2)
        # note3 = 0.4 * np.sin(times * 2.0 * np.pi * F3)
        # lc_1 = np.full(len(times), 1)
        # lc_2 = np.full(len(times), 2)
        # lc_3 = np.full(len(times), 3)
        #
        # audio1 = np.zeros(len(times))
        # lc1 = np.zeros(len(times))
        # audio1[:int(len(times) / 3)] = note1[:int(len(times) / 3)]
        # audio1[int(len(times) / 3):int(len(times) * 2 / 3)] = note2[int(len(times) / 3):int(len(times) * 2 / 3)]
        # audio1[int(len(times) * 2 / 3):] = note3[int(len(times) * 2 / 3):]
        # lc1[:int(len(times) / 3)] = lc_1[:int(len(times) / 3)]
        # lc1[int(len(times) / 3):int(len(times) * 2 / 3)] = lc_2[int(len(times) / 3):int(len(times) * 2 / 3)]
        # lc1[int(len(times) * 2 / 3):] = lc_3[int(len(times) * 2 / 3):]
        #
        #
        # audio2 = np.zeros(len(times))
        # lc2 = np.zeros(len(times))
        # audio2[:int(len(times) / 3)] = note3[:int(len(times) / 3)]
        # audio2[int(len(times) / 3):int(len(times) * 2 / 3)] = note2[int(len(times) / 3):int(len(times) * 2 / 3)]
        # audio2[int(len(times) * 2 / 3):] = note1[int(len(times) * 2 / 3):]
        # lc2[:int(len(times) / 3)] = lc_3[:int(len(times) / 3)]
        # lc2[int(len(times) / 3):int(len(times) * 2 / 3)] = lc_2[int(len(times) / 3):int(len(times) * 2 / 3)]
        # lc2[int(len(times) * 2 / 3):] = lc_1[int(len(times) * 2 / 3):]
        #
        # audio3 = np.zeros(len(times))
        # lc3 = np.zeros(len(times))
        # audio3[:int(len(times) / 3)] = note2[:int(len(times) / 3)]
        # audio3[int(len(times) / 3):int(len(times) * 2 / 3)] = note1[int(len(times) / 3):int(len(times) * 2 / 3)]
        # audio3[int(len(times) * 2 / 3):] = note3[int(len(times) * 2 / 3):]
        # lc3[:int(len(times) / 3)] = lc_2[:int(len(times) / 3)]
        # lc3[int(len(times) / 3):int(len(times) * 2 / 3)] = lc_1[int(len(times) / 3):int(len(times) * 2 / 3)]
        # lc3[int(len(times) * 2 / 3):] = lc_3[int(len(times) * 2 / 3):]
        #
        #
        # amplitudes[0, :] = audio2
        # amplitudes[1, :] = audio1
        # amplitudes[2, :] = audio3
        #
        # lc[0, :] = lc1
        # lc[1, :] = lc2
        # lc[2, :] = lc3
        #
        # return amplitudes, gc, lc



        # random.seed(100)
        # np.random.seed(100)
        if generate_two_waves:
            sample_num = 20
        else:
            sample_num = 10

        # sample_num = 1

        # audio, current_lc, current_gc = make_training_data(noisy)
        amplitudes = np.zeros((batch_size*sample_num, len(times)))
        lc = np.zeros((batch_size*sample_num, len(times)))
        gc = np.zeros(batch_size*sample_num)

        amplitudes = np.zeros((sample_num, batch_size, len(times) + receptive_field-1))
        lc = np.zeros((sample_num, batch_size, len(times) + receptive_field-1))
        gc = np.zeros((sample_num, batch_size))

        # amplitudes[:,0] = audio
        # lc[:,0] = current_lc
        # lc = np.stack(current_lc)
        # speaker_ids = np.stack([0])
        duration_lists = []
        for i in range(sample_num):
            if i < 30:
                for j in range(batch_size):
                    audio, current_lc, current_gc, duration_list  = make_training_data(noisy)
                    amplitudes[i, :] = audio
                    lc[i, :] = current_lc
                    gc[i] = current_gc
                    duration_lists.append(duration_list)
            else:
                audio, current_lc, current_gc, duration_list = make_training_data(noisy, True)
                amplitudes[i, :] = audio
                lc[i, :] = current_lc
                gc[i] = current_gc
            # amplitudes = np.stack(amplitudes, audio)
            # lc = np.stack(lc, current_lc)
            # speaker_ids = np.stack(speaker_ids, [0])


    else:
        amplitudes = (np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F3) / 3.0)
        gc = None

    return amplitudes, gc, lc, duration_lists

def make_training_data(noisy=None, wave_type=None):
    sample_period = 1.0 / SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)
    # LEADING_SILENCE = random.randint(10, 128)
    # start_time = LEADING_SILENCE / SAMPLE_RATE_HZ
    # times = times[LEADING_SILENCE:] - start_time

    # amplitude = np.zeros(shape=(len(times)))
    amplitude = np.zeros(shape=(3, len(times)))
    lc = np.zeros(shape=(3, len(times)))
    note0 = np.zeros(shape=(len(times)))
    if wave_type is None:
        note1 = 0.6 * np.sin(times * 2.0 * np.pi * F1)
        note2 = 0.5 * np.sin(times * 2.0 * np.pi * F2)
        note3 = 0.4 * np.sin(times * 2.0 * np.pi * F3)
    else:
        note1 = 0.6 * signal.sawtooth(times * 2.0 * np.pi * F1)
        note2 = 0.5 * signal.sawtooth(times * 2.0 * np.pi * F2)
        note3 = 0.4 * signal.sawtooth(times * 2.0 * np.pi * F3)
    lc_0 = np.full(len(times), 0)
    lc_1 = np.full(len(times), 1)
    lc_2 = np.full(len(times), 2)
    lc_3 = np.full(len(times), 3)
    # amplitudes[0:LEADING_SILENCE] = 0.0
    current_time = 0
    notes = {0: [note0, lc_0], 1: [note1, lc_1], 2: [note2, lc_2], 3: [note3, lc_3]}
    duration_list = []
    while True:
        _note1 = notes[random.randint(0, 3)]
        _note2 = notes[random.randint(0, 3)]
        _note3 = notes[random.randint(0, 3)]
        duration = random.randint(100, 300)
        if noisy is not None:
            noise = (np.random.rand(duration) * 2 - 1) / 10
        else:
            noise = np.zeros(duration)
        # total += duration
        if current_time + duration > len(times):
            amplitude[0, current_time:] = _note1[0][current_time:]
            lc[0, current_time:] = _note1[1][current_time:]
            amplitude[1, current_time:] = _note2[0][current_time:]
            lc[1, current_time:] = _note2[1][current_time:]
            amplitude[2, current_time:] = _note3[0][current_time:]
            lc[2, current_time:] = _note3[1][current_time:]
            break
        amplitude[0, current_time:current_time + duration] = _note1[0][current_time:current_time + duration] + noise
        lc[0, current_time:current_time + duration] = _note1[1][current_time:current_time + duration]
        amplitude[1, current_time:current_time + duration] = _note2[0][current_time:current_time + duration] + noise
        lc[1, current_time:current_time + duration] = _note2[1][current_time:current_time + duration]
        amplitude[2, current_time:current_time + duration] = _note3[0][current_time:current_time + duration] + noise
        lc[2, current_time:current_time + duration] = _note3[1][current_time:current_time + duration]
        current_time += duration
        duration_list.append(current_time)

    gc = 0 if wave_type is None else 1
    amplitude = np.pad(amplitude, ((0, 0), (receptive_field - 1, 0)), 'constant')
    lc = np.pad(lc, ((0, 0), (receptive_field - 1, 0)), 'constant')
    return amplitude, lc, gc, duration_list

def check_logits(sess, net, global_condition, local_condition):
    samples_placeholder = tf.placeholder(tf.int32)
    gc_placeholder = tf.placeholder(tf.int32) if global_condition is not None \
        else None
    lc_placeholder = tf.placeholder(tf.int32) if local_condition is not None \
        else None

    net.batch_size = 1
    fixed_wave = np.random.rand(GENERATE_SAMPLES) * 2 - 1


    """slow generation"""
    next_sample_probs = net.predict_proba(samples_placeholder,
                                          gc_placeholder, lc_placeholder, isTest=True)
    operations = [next_sample_probs]
    logits_slow = [None]
    logits_slow = gen_logits(
        sess, net, False, global_condition, local_condition, samples_placeholder,
        gc_placeholder, lc_placeholder, operations, fixed_wave)

    """fast generation"""
    next_sample_probs = net.predict_proba_incremental(samples_placeholder,
                                                      gc_placeholder, lc_placeholder, isTest=True)
    sess.run(net.init_ops)
    operations = [next_sample_probs]
    operations.extend(net.push_ops)
    logits_fast = [None]
    logits_fast = gen_logits(
        sess, net, True, global_condition, local_condition, samples_placeholder,
        gc_placeholder, lc_placeholder, operations, fixed_wave)

    # return proba_fast, logits_fast, proba_slow, logits_slow
    return logits_fast, logits_slow

def compare_logits(sess, net, gc, lc):
    logits_fast = []
    logits_slow = []
    samples_placeholder = tf.placeholder(tf.int32)
    gc_placeholder = tf.placeholder(tf.int32)
    lc_placeholder = tf.placeholder(tf.int32)

    net.batch_size = 1

    fixed_wave = np.random.rand(GENERATE_SAMPLES) * 2 - 1

    operation_fast = net.predict_proba_incremental(samples_placeholder,
                                                   gc_placeholder, lc_placeholder, isTest=True)
    operation_slow = net.predict_proba(samples_placeholder,
                                          gc_placeholder, lc_placeholder, isTest=True)

    initial_waveform = [128] * net.receptive_field

    print("step1")

    """ fast generation preparation """
    for sample in initial_waveform[:-1]:
        feed_dict_fast = {samples_placeholder: [sample]}
        feed_dict_fast[gc_placeholder] = gc
        feed_dict_fast[lc_placeholder] = [0]
        sess.run(operation_fast, feed_dict=feed_dict_fast)

    print("step2")

    for i in range(GENERATE_SAMPLES):
        if i % 100 == 0:
            print("Generating {} of {}.".format(i, GENERATE_SAMPLES))
            sys.stdout.flush()

        current_lc = lc[i]

        """ fast generation """
        if i == 0:
            window_fast = initial_waveform[-1]
        else:
            window_fast = fixed_wave[i-1]
        feed_dict_fast = {samples_placeholder: window_fast, lc_placeholder: current_lc, gc_placeholder: gc}
        logit_fast = sess.run(operation_fast, feed_dict=feed_dict_fast)

        """ slow generation """
        if i == 0:
            window_slow = initial_waveform
        else:
            window_slow = window_slow[1:]
            window_slow.append(fixed_wave[i - 1])
        feed_dict_slow = {samples_placeholder: window_slow, lc_placeholder: current_lc, gc_placeholder: gc}
        logit_slow = sess.run(operation_slow, feed_dict=feed_dict_slow)

        np.save("../data/logit_fast", logits_fast)
        np.save("../data/logit_slow", logits_slow)
        print("finished checking")
        exit()
        logits_fast.append(logit_fast)
        logits_slow.append(logit_slow)

        # sample = np.random.choice(
        #     np.arange(QUANTIZATION_CHANNELS), p=proba[0])
        # waveform.append(sample)


def gen_logits(sess, net, fast_generation, gc, lc, samples_placeholder,
                      gc_placeholder, lc_placeholder, operations, fixed_wave):

    logits = []
    waveform = []
    initial_waveform = [128] * net.receptive_field

    # initial_lc = [0] * net.receptive_field if lc is not None else None
    if fast_generation:
        for sample in initial_waveform[:-1]:
            feed_dict = {samples_placeholder: [sample]}
            feed_dict[gc_placeholder] = gc if gc is not None else None
            feed_dict[lc_placeholder] = [1] if lc is not None else None
            sess.run(operations, feed_dict)

    for i in range(GENERATE_SAMPLES):
        if i % 100 == 0:
            print("Generating {} of {}.".format(i, GENERATE_SAMPLES))
            sys.stdout.flush()
        if fast_generation:
            if i == 0:
                window = initial_waveform[-1]
            else:
                window = fixed_wave[i-1]
        else:
            if i == 0:
                window = initial_waveform
            else:
                window = window[1:]
                window.append(fixed_wave[i - 1])

        current_lc = lc[i] if lc is not None else None
        # Run the WaveNet to predict the next sample.
        feed_dict = {samples_placeholder: window, lc_placeholder: current_lc}
        if gc is not None:
            feed_dict[gc_placeholder] = gc

        logit = sess.run(operations, feed_dict=feed_dict)

        logits.append(logit)
        # sample = np.random.choice(
        #     np.arange(QUANTIZATION_CHANNELS), p=proba[0])
        # waveform.append(sample)
    # print("logit shape")
    # print(logit.shape)

    # return waveform, logits
    return logits


def generate_waveform(sess, net, fast_generation, gc, lc, samples_placeholder,
                      gc_placeholder, lc_placeholder, operations, test=False, fixed_wave=None):
    if test:
        logits = []

    waveform = [128] * net.receptive_field

    # initial_lc = [0] * net.receptive_field if lc is not None else None
    if fast_generation:
        for sample in waveform[:-1]:
            feed_dict = {samples_placeholder: [sample]}
            feed_dict[gc_placeholder] = gc if gc is not None else None
            feed_dict[lc_placeholder] = [0] if lc is not None else None
            sess.run(operations, feed_dict)
            if fixed_wave is not None:
                np.insert(fixed_wave,0,0)

    for i in range(GENERATE_SAMPLES):
        if i % 100 == 0:
            print("Generating {} of {}.".format(i, GENERATE_SAMPLES))
            sys.stdout.flush()
        if fast_generation:
            window = waveform[-1]
            current_lc = lc[i] if lc is not None else None
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
                current_lc = lc[i] if lc is not None else None
            else:
                window = waveform
                # current_lc = initial_lc if lc is not None else None
                current_lc = [0] if lc is not None else None
            # current_lc = current_lc.reshape((1,))
            # print("current lc")
            # print(current_lc.shape)
            # print(gc.shape)

        # Run the WaveNet to predict the next sample.
        feed_dict = {samples_placeholder: window, lc_placeholder: current_lc}
        if gc is not None:
            feed_dict[gc_placeholder] = gc

        # if lc is not None:
        #     feed_dict[lc_placeholder] = current_lc
        results = sess.run(operations, feed_dict=feed_dict)

        if test:
            logits.append(results)
        else:
            sample = np.random.choice(
               np.arange(QUANTIZATION_CHANNELS), p=results[0])
            waveform.append(sample)

    # Skip the first number of samples equal to the size of the receptive
    # field minus one.
    if test:
        return logits
    else:
        waveform = np.array(waveform[net.receptive_field - 1:])
        decode = mu_law_decode(samples_placeholder, QUANTIZATION_CHANNELS)
        decoded_waveform = sess.run(decode,
                                    feed_dict={samples_placeholder: waveform})
        return decoded_waveform


def generate_waveforms(sess, net, fast_generation, global_condition, local_condition, test=False):
    samples_placeholder = tf.placeholder(tf.int32)
    gc_placeholder = tf.placeholder(tf.int32) if global_condition is not None \
        else None
    lc_placeholder = tf.placeholder(tf.int32) if local_condition is not None \
        else None


    net.batch_size = 1

    if fast_generation:
        next_sample_probs = net.predict_proba_incremental(samples_placeholder,
                                                          gc_placeholder, lc_placeholder)
        sess.run(net.init_ops)
        operations = [next_sample_probs]
        operations.extend(net.push_ops)
    else:
        next_sample_probs = net.predict_proba(samples_placeholder,
                                              gc_placeholder, lc_placeholder)
        operations = [next_sample_probs]

    num_waveforms = 1 if global_condition is None else  \
        global_condition.shape[0]
    gc = None
    lc = None
    waveforms = [None] * num_waveforms
    for waveform_index in range(num_waveforms):
        if global_condition is not None:
            # gc = global_condition[waveform_index, :]
            gc = global_condition[waveform_index]
        if local_condition is not None:
            # gc = global_condition[waveform_index, :]
            lc = local_condition
        # Generate a waveform for each speaker id.
        print("Generating waveform {}.".format(waveform_index))
        waveforms[waveform_index] = generate_waveform(
            sess, net, fast_generation, gc, lc, samples_placeholder,
            gc_placeholder, lc_placeholder, operations)

    return waveforms, global_condition


def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


def check_waveform(assertion, generated_waveform, gc_category):
    # librosa.output.write_wav('/tmp/sine_test{}.wav'.format(gc_category),
    #                          generated_waveform,
    #                          SAMPLE_RATE_HZ)
    power_spectrum = np.abs(np.fft.fft(generated_waveform))**2
    freqs = np.fft.fftfreq(generated_waveform.size, SAMPLE_PERIOD_SECS)
    indices = np.argsort(freqs)
    indices = [index for index in indices if freqs[index] >= 0 and
               freqs[index] <= 500.0]
    power_spectrum = power_spectrum[indices]
    freqs = freqs[indices]
    # plt.plot(freqs[indices], power_spectrum[indices])
    # plt.show()
    power_sum = np.sum(power_spectrum)
    f1_power = find_nearest(freqs, power_spectrum, F1)
    f2_power = find_nearest(freqs, power_spectrum, F2)
    f3_power = find_nearest(freqs, power_spectrum, F3)
    if gc_category is None:
        # We are not globally conditioning to select one of the three sine
        # waves, so expect it across all three.
        expected_power = f1_power + f2_power + f3_power
        assertion(expected_power, 0.7 * power_sum)
    else:
        # We expect spectral power at the selected frequency
        # corresponding to the gc_category to be much higher than at the other
        # two frequencies.
        frequency_lut = {0: f1_power, 1: f2_power, 2: f3_power}
        other_freqs_lut = {0: f2_power + f3_power,
                           1: f1_power + f3_power,
                           2: f1_power + f2_power}
        expected_power = frequency_lut[gc_category]
        # Power at the selected frequency should be at least 10 times greater
        # than at other frequences.
        # This is a weak criterion, but still detects implementation errors
        # in the code.
        assertion(expected_power, 10.0*other_freqs_lut[gc_category])

def plot_waveform(waveform):
    power_spectrum = np.abs(np.fft.fft(waveform))
    freqs = np.fft.fftfreq(np.shape(waveform)[0], 1.0 / SAMPLE_RATE_HZ)
    indices = np.argsort(freqs)
    margin = 50
    indices = [i for i in indices if (min(NOTES_HZ) - margin <= freqs[i] <= max(NOTES_HZ) + margin)]
    plt.subplot(2, 1, 1)
    plt.ylabel('Power')
    plt.xlabel('Frequency [Hz]')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.plot(freqs[indices], power_spectrum[indices])
    plt.subplot(2, 1, 2)
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.plot(waveform)
    plt.show()

def plot_waveform4eachLC(audio, lc):
    index_start = 0
    # index_end = 0
    current_lc = -1
    for i, l in enumerate(lc):
        if current_lc != l:
            if current_lc != -1:
                plot_waveform(audio[index_start:i])
                current_lc = l
            else:
                current_lc = l
            index_start = i
    plot_waveform(audio[index_start:])


class TestNet(tf.test.TestCase):
    def setUp(self):
        print('TestNet setup.')
        sys.stdout.flush()

        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.local_conditioning = False
        self.train_iters = TRAIN_ITERATIONS
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                skip_channels=32,
                                global_condition_channels=None,
                                global_condition_cardinality=None)

    def _save_net(sess):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(sess, os.path.join('tmp', 'test.ckpt'))

    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.




    def testEndToEndTraining(self):
        def shuffle_row(audio, gc, lc):
            from copy import deepcopy
            for i in range(10):
                index1 = random.randint(0, audio.shape[0] - 1)
                index2 = random.randint(0, audio.shape[0] - 1)
                audio1 = deepcopy(audio[index1, :])
                audio2 = deepcopy(audio[index2, :])
                audio[index1, :] = audio2
                audio[index2, :] = audio1
                lc1 = deepcopy(lc[index1, :])
                lc2 = deepcopy(lc[index2, :])
                lc[index1, :] = lc2
                lc[index2, :] = lc1
                gc1 = deepcopy(gc[index1])
                gc2 = deepcopy(gc[index2])
                gc[index1] = gc2
                gc[index2] = gc1
            return audio, gc, lc

        def CreateTrainingFeedDict(audio, gc, lc, audio_placeholder,
                                   gc_placeholder, lc_placeholder, i):
            speaker_index = 0

            i = i % int(audio.shape[0]/self.net.batch_size)
            if i==0:
                audio, gc, lc = shuffle_row(audio, gc, lc)
            _audio = audio[i*self.net.batch_size:(i+1)*self.net.batch_size]
            _gc = gc[i*self.net.batch_size:(i+1)*self.net.batch_size]
            _lc = lc[i*self.net.batch_size:(i+1)*self.net.batch_size]
            print("training audio length")
            print(_audio.shape)
            exit()

            if gc is None:
                # No global conditioning.
                feed_dict = {audio_placeholder: _audio}
            elif self.global_conditioning and not self.local_conditioning:
                feed_dict = {audio_placeholder: _audio,
                             gc_placeholder: _gc}
            elif not self.global_conditioning and self.local_conditioning:
                feed_dict = {audio_placeholder: _audio,
                             lc_placeholder: _lc}
            elif self.global_conditioning and self.local_conditioning:
                feed_dict = {audio_placeholder: _audio,
                             gc_placeholder: _gc,
                             lc_placeholder: _lc}
            return feed_dict, speaker_index, audio, gc, lc


        np.random.seed(42)
        receptive_field = self.net.receptive_field
        audio, gc, lc, duration_lists = make_sine_waves(self.global_conditioning, self.local_conditioning, True)
        waveform_size = audio.shape[1]

        print("shape check 1")
        print(audio.shape)
        print(gc.shape)
        print(lc.shape)
        # Pad with 0s (silence) times size of the receptive field minus one,
        # because the first sample of the training data is 0 and if the network
        # learns to predict silence based on silence, it will generate only
        # silence.
        # if self.global_conditioning:
        #     # print(audio.shape)
        #     audio = np.pad(audio, ((0, 0), (self.net.receptive_field - 1, 0)), 'constant')
        #     # lc = np.pad(lc, ((0,0), (self.net.receptive_field - 1, 0)), 'maximum')
        #     # to set lc=0 for the initial silence
        #     lc = np.pad(lc, ((0, 0), (self.net.receptive_field - 1, 0)), 'constant')
        #     # print(audio.shape)
        #     # exit()
        # else:
        #     # print(audio.shape)
        #     audio = np.pad(audio, (self.net.receptive_field - 1, 0),
        #                    'constant')
            # print(audio.shape)
            # exit()

        audio_placeholder = tf.placeholder(dtype=tf.float32)
        gc_placeholder = tf.placeholder(dtype=tf.int32)  \
            if self.global_conditioning else None
        lc_placeholder = tf.placeholder(dtype=tf.int32) \
            if self.local_conditioning else None

        loss = self.net.loss(input_batch=audio_placeholder,
                             global_condition_batch=gc_placeholder,
                             local_condition_batch=lc_placeholder)
        validation = self.net.validation(input_batch=audio_placeholder,
                             global_condition_batch=gc_placeholder,
                             local_condition_batch=lc_placeholder)
        optimizer = optimizer_factory[self.optimizer_type](
                      learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.global_variables_initializer()

        generated_waveform = None
        max_allowed_loss = 0.1
        loss_val = max_allowed_loss
        initial_loss = None
        operations = [loss, optim, validation]
        with self.test_session() as sess:
            # feed_dict, speaker_index, audio, gc, lc  = CreateTrainingFeedDict(
            #     audio, gc, lc, audio_placeholder, gc_placeholder, lc_placeholder, 0)
            sess.run(init)
            # print("shape check 2")
            # print(audio.shape)
            # print(lc.shape)
            # print(gc.shape)
            # print(feed_dict[audio_placeholder].shape)
            # print(feed_dict[gc_placeholder].shape)
            # print(feed_dict[lc_placeholder].shape)
            # initial_loss = sess.run(loss, feed_dict=feed_dict)
            _gc = np.zeros(3)
            for i in range(self.train_iters):
                # for lc_index in range(3):
                #     current_audio = audio[:, int(lc_index * (waveform_size / 3)): int(
                #         (lc_index + 1) * (waveform_size / 3) + self.net.receptive_field)]
                #     # print(current_audio.shape)
                #     current_lc = lc[:, int(lc_index * (waveform_size / 3)): int(
                #         (lc_index + 1) * (waveform_size / 3) + self.net.receptive_field)]
                #
                #     [results] = sess.run([operations],
                #                          feed_dict={audio_placeholder: current_audio, lc_placeholder: current_lc,
                #                                     gc_placeholder: gc})
                a = 0
                current_audio = audio[i % 10]
                current_lc = lc[i % 10]
                duration_list = duration_lists[i % 10]
                start_time = 0
                for duration in duration_list:
                    _audio = current_audio[:, start_time:duration+receptive_field]
                    _lc = current_lc[:, start_time:duration+receptive_field]
                    start_time = duration

                    [results] = sess.run([operations],
                                     feed_dict={audio_placeholder: _audio, lc_placeholder: _lc,
                                                gc_placeholder: _gc})
                # feed_dict, speaker_index, audio, gc, lc = CreateTrainingFeedDict(
                #     audio, gc, lc, audio_placeholder, gc_placeholder, lc_placeholder, i)
                # [results] = sess.run([operations], feed_dict=feed_dict)
                    if i % 100 == 0:
                        print("i: %d loss: %f" % (i, results[0]))
                        print("i: %d validation: %f" % (i, results[0]))

            loss_val = results[0]

            # Sanity check the initial loss was larger.
            # self.assertGreater(initial_loss, max_allowed_loss)

            # Loss after training should be small.
            # self.assertLess(loss_val, max_allowed_loss)

            # Loss should be at least two orders of magnitude better
            # than before training.
            # self.assertLess(loss_val / initial_loss, 0.02)


            if self.generate:
                # self._save_net(sess)
                if self.global_conditioning and not self.local_conditioning:
                    # Check non-fast-generated waveform.
                    generated_waveforms, ids = generate_waveforms(
                        # sess, self.net, True, speaker_ids)
                        sess, self.net, True, np.array((0,)))
                    for (waveform, id) in zip(generated_waveforms, ids):
                        # check_waveform(self.assertGreater, waveform, id[0])
                        check_waveform(self.assertGreater, waveform, id)

                elif self.global_conditioning and self.local_conditioning:
                    lc_0 = np.full(int(GENERATE_SAMPLES / 3), 1)
                    lc_1 = np.full(int(GENERATE_SAMPLES / 3), 2)
                    lc_2 = np.full(int(GENERATE_SAMPLES / 3), 3)
                    lc = np.concatenate((lc_0, lc_1, lc_2))
                    lc = lc.reshape((lc.shape[0],1))
                    print(lc.shape)
                    """ * test * """
                    test = False
                    if test:
                        # compare_logits(sess, self.net, np.array((0,)), lc)
                        logits_fast, logits_slow = check_logits(sess, self.net, np.array((0,)), lc)
                        np.save("../data/logits_fast", logits_fast)
                        np.save("../data/logits_slow", logits_slow)
                        # np.save("../data/proba_fast", proba_fast)
                        # np.save("../data/proba_slow", proba_slow)
                        exit()
                    # Check non-fast-generated waveform.
                    if self.generate_two_waves:
                        generated_waveforms, ids = generate_waveforms(
                            sess, self.net, True, np.array((0,1)), lc)
                    else:
                        generated_waveforms, ids = generate_waveforms(
                            sess, self.net, True, np.array((0,)), lc)
                    for (waveform, id) in zip(generated_waveforms, ids):
                        # check_waveform(self.assertGreater, waveform, id[0])
                        if id == 0:
                            np.save("../data/wave_fast", waveform)
                            np.save("../data/lc_fast", lc)
                            # plot_waveform(waveform)
                        else:
                            np.save("../data/wave_t", waveform)

                    generated_waveforms, ids = generate_waveforms(
                        sess, self.net, False, np.array((0,)), lc)

                    for (waveform, id) in zip(generated_waveforms, ids):
                        # check_waveform(self.assertGreater, waveform, id[0])
                        if id == 0:
                            np.save("../data/wave_slow", waveform)
                            np.save("../data/lc_slow", lc)
                            # plot_waveform(waveform)
                        else:
                            np.save("../data/wave_t", waveform)
                            # np.save("../data/lc", lc)
                        # plot_waveform4eachLC(waveform, lc)
                        # check_waveform(self.assertGreater, waveform, id)

                    # Check fast-generated wveform.
                    # generated_waveforms, ids = generate_waveforms(sess,
                    #     self.net, True, speaker_ids)
                    # for (waveform, id) in zip(generated_waveforms, ids):
                    #     print("Checking fast wf for id{}".format(id[0]))
                    #     check_waveform( self.assertGreater, waveform, id[0])

                else:
                    # Check non-incremental generation
                    generated_waveforms, _ = generate_waveforms(
                        sess, self.net, False, None)
                    check_waveform(
                        self.assertGreater, generated_waveforms[0], None)
                    # Check incremental generation
                    generated_waveform = generate_waveforms(
                        sess, self.net, True, None)
                    check_waveform(
                        self.assertGreater, generated_waveforms[0], None)


class TestNetWithBiases(TestNet):

    def setUp(self):
        print('TestNetWithBias setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=32)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.train_iters = TRAIN_ITERATIONS


class TestNetWithRMSProp(TestNet):

    def setUp(self):
        print('TestNetWithRMSProp setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                skip_channels=256)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = True
        self.momentum = MOMENTUM
        self.train_iters = TRAIN_ITERATIONS
        self.global_conditioning = False


class TestNetWithScalarInput(TestNet):

    def setUp(self):
        print('TestNetWithScalarInput setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=32,
                                scalar_input=True,
                                initial_filter_width=4)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.01
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.train_iters = 1000


class TestNetWithGlobalConditioning(TestNet):
    def setUp(self):
        print('TestNetWithGlobalConditioning setup.')
        sys.stdout.flush()

        self.optimizer_type = 'sgd'
        self.learning_rate = 0.01
        self.generate = True
        self.momentum = MOMENTUM
        self.global_conditioning = True
        self.train_iters = 500
        self.net = WaveNetModel(batch_size=NUM_SPEAKERS,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=256,
                                global_condition_channels=NUM_SPEAKERS,
                                global_condition_cardinality=NUM_SPEAKERS)


class TestNetWithLocalConditioning(TestNet):
    def setUp(self):
        print('TestNetWithGlobalConditioning setup.')
        sys.stdout.flush()

        self.optimizer_type = 'sgd'
        self.learning_rate = 0.01
        self.generate = True
        self.momentum = MOMENTUM
        self.global_conditioning = True
        self.local_conditioning = True
        self.training_data_random = True
        self.training_data_noisy = False
        self.generate_two_waves = False

        self.train_iters = 1001


        self.net = WaveNetModel(batch_size=3,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=256,
                                global_condition_channels=NUM_SPEAKERS,
                                global_condition_cardinality=NUM_SPEAKERS,
                                local_condition_channels=128,
                                local_condition_cardinality=NUM_SPEAKERS+1)

if __name__ == '__main__':
    tf.test.main()
