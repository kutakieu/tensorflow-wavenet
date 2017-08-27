import fnmatch
import os
import random
import re
import threading
import subprocess

import librosa
import numpy as np
import tensorflow as tf

from pytube import YouTube
import moviepy.editor as mp
from PIL import Image
from .vectorise_image import image2vector


FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id

def load_generic_audio_video(directory, sample_rate):

    # download_youtube(directory)
    clip = mp.VideoFileClip(directory + "/tmp.mp4")
    # clip.audio.write_audiofile(directory + "/tmp.wav")

    audio, _ = librosa.load(directory + "/tmp.wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    i2v = image2vector([32, 18, 3])

    sample_size = int(sample_rate / clip.fps + 0.5)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    num_frames = int(clip.duration * clip.fps) - 1
    for i in range(num_frames):
        img = Image.fromarray(clip.get_frame(i))
        img.thumbnail([32, 18], Image.ANTIALIAS)
        img = np.array(img) / 255
        h, w = img.shape[0], img.shape[1]
        img = img.reshape((1, w, h, 3))
        image_vector = i2v.convert(img)
        image_vector = image_vector.reshape(512, 1)
        image_vectors = np.tile(image_vector, sample_size)
        # yield a set of data for each frame and corresponding audio data
        yield audio[i*sample_size : (i+1)*sample_size], image_vectors


# def convert_video2vector(directory, sample_rate):



def download_youtube(directory, video_name=None):
    subprocess.call(["rm", "tmp.wav", "tmp.mp4"])

    video_id = "h6yJEHHT5eA"
    try:
        youtube = YouTube("https://www.youtube.com/watch?v=" + video_id)
        youtube.set_filename('tmp')
    except:
        print("there is no video")

    try:
        video = youtube.get('mp4', '360p')
    except:
        print("there is no video for this setting")

    video.download(directory)




def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 lc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.lc_enabled = lc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        if self.lc_enabled:
            self.lc_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 512))
            self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                                shapes=[(None, 512)])
            self.lc_enqueue = self.lc_queue.enqueue([self.lc_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.

        if audio_dir is not None:
            files = find_files(audio_dir)
            """
            if not files:
                raise ValueError("No audio files found in '{}'.".format(audio_dir))
            if self.gc_enabled and not_all_have_id(files):
                raise ValueError("Global conditioning is enabled, but file names "
                                 "do not conform to pattern having id.")
            """
            # Determine the number of mutually-exclusive categories we will
            # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def dequeue_lc(self, num_elements):
        return self.lc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio_video(self.audio_dir, self.sample_rate)
            # for audio, filename, category_id, video_vectors in iterator:
            for audio, video_vectors in iterator:
                self.sample_size = len(audio)
                if self.coord.should_stop():
                    stop = True
                    break

                # ignore trim silence for now
                # if self.silence_threshold is not None:
                #     # Remove silence
                #     audio = trim_silence(audio[:, 0], self.silence_threshold)
                #     audio = audio.reshape(-1, 1)
                #     if audio.size == 0:
                #         print("Warning: {} was ignored as it contains only "
                #               "silence. Consider decreasing trim_silence "
                #               "threshold, or adjust volume of the audio."
                #               .format(filename))

                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')
                # pad the video vector
                pad = np.zeros((512, self.receptive_field))
                video_vectors = np.concatenate((pad, video_vectors),axis=1)

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        # print("hogehogehoeg")
                        # print(piece.shape)
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                        # gc is not available for now
                        # if self.gc_enabled:
                        #     sess.run(self.gc_enqueue, feed_dict={
                        #         self.id_placeholder: category_id})
                        if self.lc_enabled:
                            piece = video_vectors[:(self.receptive_field +
                                            self.sample_size), :]
                            piece = piece.transpose()
                            sess.run(self.lc_enqueue, feed_dict={
                                self.lc_placeholder: piece})
                            # TODO implement here
                            video_vectors = video_vectors[self.sample_size:, :]

                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    # if self.gc_enabled:
                    #     sess.run(self.gc_enqueue,
                    #              feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads


def main():
    print("start download youtube")
    download_youtube()
    print("done!")

if __name__ == '__main__':
    main()