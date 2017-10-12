"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
import numpy as np
import librosa
import pickle

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, AudioReader, optimizer_factory, audio_reader, mu_law_decode

BATCH_SIZE = 1
# DATA_DIRECTORY = './VCTK-Corpus'
DATA_DIRECTORY = "./data/"
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 1000
NUM_STEPS = int(1e3)
# NUM_STEPS = int(5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0.00005
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False
SAMPLE_RATE = 16000

isDebug = True


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--lc_channels', type=int, default=None,
                        help='Number of local condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    parser.add_argument('--restore_model', type=str, default=None,
                        help='Restore the trained model to restart training: path to the model')
    parser.add_argument('--isDebug', type=str, default="False",
                        help='Run this program to debug or not')
    parser.add_argument('--generate_every', type=int, default=5,
                        help='How many steps to calculate validation score and generate sound file after.')
    parser.add_argument('--lstm_len', type=int, default=24,
                        help='The length of the input for the LSTM cell.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def prediction2sample(prediction, temperature, quantization_channels):
    # Scale prediction distribution using temperature.

    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = (scaled_prediction -
                         np.logaddexp.reduce(scaled_prediction))
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')

    # Prediction distribution at temperature=1.0 should be unchanged after
    # scaling.
    # if temperature == 1.0:
    #     np.testing.assert_allclose(
    #         prediction, scaled_prediction, atol=1e-5,
    #         err_msg='Prediction scaling at temperature=1.0 '
    #                 'is not working as intended.')

    sample = np.random.choice(
        np.arange(quantization_channels), p=scaled_prediction)
    return sample

def main():
    args = get_arguments()

    if args.isDebug in ["True", "true", "t", "1"]:
        isDebug = True
        print("Running train.py for debugging...")
    elif args.isDebug in ["False", "false", "f", "0"]:
        isDebug = False
        print("Running train.py for actual training...")
    else:
        print("--isDebug has to be True or False")
        exit()


    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']
    print(restore_from)

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()


    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        lc_enabled = args.lc_channels is not None
        # reader = AudioReader(
        #     args.data_dir,
        #     coord,
        #     sample_rate=wavenet_params['sample_rate'],
        #     gc_enabled=gc_enabled,
        #     lc_enabled=lc_enabled,
        #     receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
        #                                                            wavenet_params["dilations"],
        #                                                            wavenet_params["scalar_input"],
        #                                                            wavenet_params["initial_filter_width"]),
        #     sample_size=args.sample_size,
        #     silence_threshold=silence_threshold)
        # audio_batch = reader.dequeue(args.batch_size)
        # if gc_enabled:
        #     gc_id_batch = reader.dequeue_gc(args.batch_size)
        # else:
        #     gc_id_batch = None
        #
        # if lc_enabled:
        #     lc_id_batch = reader.dequeue_lc(args.batch_size)
        # else:
        #     lc_id_batch = None

    # Create network.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
        histograms=args.histograms,
        global_condition_channels=args.gc_channels,
        # global_condition_cardinality=reader.gc_category_cardinality,
        local_condition_channels=args.lc_channels,
        lstm_length=args.lstm_len)





    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None

    audio_placeholder_training = tf.placeholder(dtype=tf.float32, shape=None)
    gc_placeholder_training = tf.placeholder(dtype=tf.int32) if gc_enabled else None
    lc_placeholder_training = tf.placeholder(dtype=tf.float32,
                                               shape=(net.batch_size, net.lstm_length, 512)) if lc_enabled else None
    loss = net.loss(input_batch=audio_placeholder_training,
                    global_condition_batch=gc_placeholder_training,
                    local_condition_batch = lc_placeholder_training,
                    l2_regularization_strength=args.l2_regularization_strength)
    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    """variables for validation"""
    net.batch_size = 1
    audio_placeholder_validation = tf.placeholder(dtype=tf.float32, shape=None)
    gc_placeholder_validation = tf.placeholder(dtype=tf.int32) if gc_enabled else None
    lc_placeholder_validation = tf.placeholder(dtype=tf.float32, shape=(net.batch_size, net.lstm_length, 512)) if lc_enabled else None
    validation = net.validation(input_batch=audio_placeholder_validation,
                    global_condition_batch=gc_placeholder_validation,
                    local_condition_batch = lc_placeholder_validation)

    net.batch_size = args.batch_size


    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    # if args.restore_model is not None:
    #     variables_to_restore = {
    #         var.name[:-2]: var for var in tf.global_variables()
    #         if not ('state_buffer' in var.name or 'pointer' in var.name)}
    #     saver = tf.train.Saver(variables_to_restore)
    #
    #     print('Restoring model from {}'.format(args.checkpoint))
    #     saver.restore(sess, args.checkpoint)
    #
    #     print("Restoring model done")
    # else:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # reader.start_threads(sess)

    training_log_file = open(DATA_DIRECTORY + "training_log.txt", "w")
    validation_log_file = open(DATA_DIRECTORY + "validation_log.txt", "w")

    last_saved_step = saved_global_step

    with open('audio_lists_training.pkl', 'rb') as f1:
        audio_lists_training = pickle.load(f1)

    with open('img_vec_lists_training_lstm_5.pkl', 'rb') as f2:
        img_vec_lists_training = pickle.load(f2)

    with open('audio_lists_validation.pkl', 'rb') as f3:
        audio_lists_validation = pickle.load(f3)

    with open('img_vec_lists_validation_lstm_5.pkl', 'rb') as f4:
        img_vec_lists_validation = pickle.load(f4)



    try:
        for epoch in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()

            """ training """
            net.batch_size = args.batch_size
            # num_video_frames = []
            # training_data = audio_reader.load_generic_audio_video_without_downloading(DATA_DIRECTORY, SAMPLE_RATE,
            #                                                                             reader.i2v, "training1", args.lstm_len, net.receptive_field, num_video_frames)

            frame_index = 1

            for index in range(len(img_vec_lists_training[0])):
                audio = audio_lists_training[0][index]
                img_vec = img_vec_lists_training[0][index]
                # audio = np.pad(audio, [[net.receptive_field, 0], [0, 0]], 'constant')
                audio1 = audio_lists_training[1][index]
                img_vec1 = img_vec_lists_training[1][index]
                # audio1 = np.pad(audio1, [[net.receptive_field, 0], [0, 0]], 'constant')
                audio2 = audio_lists_training[2][index]
                img_vec2 = img_vec_lists_training[2][index]
                # audio2 = np.pad(audio2, [[net.receptive_field, 0], [0, 0]], 'constant')
                audio = np.vstack((audio, audio1))
                audio = np.vstack((audio, audio2))
                img_vec = np.vstack((img_vec, img_vec1))
                img_vec = np.vstack((img_vec, img_vec2))

                summary, loss_value, _ = sess.run([summaries, loss, optim], feed_dict={audio_placeholder_training: audio,
                                                                    lc_placeholder_training: img_vec})

                duration = time.time() - start_time
                if frame_index % 10 == 0:
                    print('epoch {:d}, frame_index {:d}/{:d} - loss = {:.3f}, ({:.3f} sec/epoch)'
                      .format(epoch, frame_index, len(img_vec_lists_training[0]), loss_value, duration))
                    training_log_file.write('epoch {:d}, frame_index {:d}/{:d} - loss = {:.3f}, ({:.3f} sec/epoch)'
                      .format(epoch, frame_index, len(img_vec_lists_training[0]), loss_value, duration))
                frame_index += 1

                if frame_index == 11 and isDebug:
                    break


            """validation and generation"""
            if epoch % args.generate_every == 0:
                print("calculating validation score...")
                net.batch_size = 1
                # num_video_frames = []
                # validation_data = audio_reader.load_generic_audio_video_without_downloading(DATA_DIRECTORY, SAMPLE_RATE,
                #                                                                             reader.i2v, "validation", args.lstm_len, num_video_frames)
                validation_score = 0

                frame_index = 1
                waveform = []

                for index in range(len(img_vec_lists_validation[0])):
                    audio = audio_lists_validation[index]
                    img_vec = img_vec_lists_validation[index]
                    # return the validation score and prediction at the same time
                    validation_value, prediction = sess.run(validation, feed_dict={audio_placeholder_validation: audio,
                                                                        lc_placeholder_validation: img_vec})

                    validation_score += validation_value

                    if prediction is not None:
                        for i in range(prediction.shape[0]):
                            # generate a sample based on the predection
                            sample = prediction2sample(prediction[i,:], 1.0, net.quantization_channels)
                            waveform.append(sample)

                    if frame_index % 10 == 0:
                        # show the progress
                        print('validation {:d}/{:d}'.format(frame_index, len(img_vec_lists_validation[0])))
                    frame_index += 1

                    if frame_index == 10 and isDebug:
                        break

                print('epoch {:d} - validation = {:.3f}'
                      .format(epoch, sum(validation_score)))
                validation_log_file.write('epoch {:d} - validation = {:.3f}\n'.format(epoch, sum(validation_score)))

                if len(waveform) > 0:
                    decode = mu_law_decode(audio_placeholder_validation, wavenet_params['quantization_channels'])
                    out = sess.run(decode, feed_dict={audio_placeholder_validation: waveform})
                    write_wav(out, wavenet_params['sample_rate'], DATA_DIRECTORY + "epoch_" + str(epoch) + ".wav")

            if epoch % args.checkpoint_every == 0:
                save(saver, sess, logdir, epoch)
                last_saved_step = epoch

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        validation_log_file.close()
        training_log_file.close()
        if epoch > last_saved_step:
            save(saver, sess, logdir, epoch)
        # coord.request_stop()
        # coord.join(threads)

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

if __name__ == '__main__':
    main()
