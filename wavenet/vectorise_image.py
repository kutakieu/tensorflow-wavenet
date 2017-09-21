import numpy as np
import tensorflow as tf
import vgg
from PIL import Image
import moviepy.editor as mp


class image2vector(object):
    def __init__(self, img_shape, vgg19_npy_path="./vgg19.npy"):
        # self.vgg19_npy_path = "./vgg19.npy"
        self.VGG = np.load(vgg19_npy_path, encoding='latin1').item()
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.img_shape = img_shape
        self.graph = tf.Graph()
        self.cnn = self.model()
        print(self.img_shape)



    def model(self):

        with self.graph.as_default():

            with tf.name_scope("cnn"):
                with tf.name_scope("input"):
                    x = tf.placeholder(tf.float32, shape=(None, self.img_shape[0], self.img_shape[1], self.img_shape[2]),name="x-input")
                with tf.variable_scope("conv1"):
                    filt = tf.constant(self.VGG["conv1_1"][0], name="filter")
                    conv = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv1_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv1_2"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv1_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    pool = tf.nn.avg_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

                with tf.variable_scope("conv2"):
                    filt = tf.constant(self.VGG["conv2_1"][0], name="filter")
                    conv = tf.nn.conv2d(pool, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv2_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv2_2"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv2_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)
                    pool = tf.nn.avg_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

                with tf.variable_scope("conv3"):
                    filt = tf.constant(self.VGG["conv3_1"][0], name="filter")
                    conv = tf.nn.conv2d(pool, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv3_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv3_2"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv3_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv3_3"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv3_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv3_4"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv3_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)
                    pool = tf.nn.avg_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

                with tf.variable_scope("conv4"):
                    filt = tf.constant(self.VGG["conv4_1"][0], name="filter")
                    conv = tf.nn.conv2d(pool, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv4_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv4_2"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv4_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv4_3"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv4_3"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv4_4"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv4_4"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)
                    pool = tf.nn.avg_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

                with tf.variable_scope("conv5"):
                    filt = tf.constant(self.VGG["conv5_1"][0], name="filter")
                    conv = tf.nn.conv2d(pool, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv5_1"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv5_2"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv5_2"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv5_3"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv5_3"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    filt = tf.constant(self.VGG["conv5_4"][0], name="filter")
                    conv = tf.nn.conv2d(relu, filt, [1, 1, 1, 1], padding='SAME')
                    conv_biases = tf.constant(self.VGG["conv5_4"][1], name="biases")
                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)
                    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

        return pool
        # shape = pool.get_shape().as_list()
        # # depth = shape[1] * shape[2] * shape[3]
        # reshape = tf.reshape(relu, [-1, shape[1] * shape[2] * shape[3]])
        #
        # print(reshape.shape)
        # return reshape

    def convert(self, img):
        rgb_scaled = img * 255.0
        red, green, blue = np.split(rgb_scaled, 3, 3)
        bgr = np.concatenate([
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2], ]
            , 3)

        with tf.Session(graph=self.graph) as sess:
            # print(self.graph.get_operations())
            # sess = tf.Session()
            # with sess.as_default():
            #     print(graph.get_operation_by_name("input/x-input"))
            x = self.graph.get_operation_by_name("cnn/input/x-input").outputs[0]
            # print(self.graph.get_operation_by_name("cnn/input/x-input"))
            # print(x)
            # print(bgr.shape)
            return sess.run(self.cnn, feed_dict={x: bgr})


#
# def main():
#     # img = Image.open("./self.jpg")
#     #
#     # img.thumbnail([224, 224], Image.ANTIALIAS)
#     # img.save("./self_resized.jpg", "JPEG")
#     # img = np.array(img) / 255
#     # print(img.shape)
#     # h, w = img.shape[0], img.shape[1]
#     # print(h)
#     # print(w)
#     # img = img.reshape((1, h, w, 3))
#     # print(img.shape)
#     # print(img.shape[1:])
#     # i2v = image2vector(img.shape[1:])
#     # vector = i2v.convert(img)
#     # print(vector.shape)
#     print("here")
#
#     import moviepy.editor as mp
#     import librosa
#     sample_rate = 16000
#     directory = "."
#
#     clip = mp.VideoFileClip(directory + "/tmp.mp4")
#     # clip.audio.write_audiofile(directory + "/tmp.wav")
#     #
#     # audio, _ = librosa.load(directory + "/tmp.wav", sr=sample_rate, mono=True)
#
#     print("step1")
#     w = 16*3
#     h = 9*3
#     i2v = image2vector([w, h, 3])
#
#     sample_size = int(sample_rate / clip.fps + 0.5)
#     img = Image.fromarray(clip.get_frame(0))
#     print("step2")
#     print(img.size)
#     img.thumbnail([w, h], Image.ANTIALIAS)
#     img.save("tmp.jpg")
#     print("step3")
#     print(img.size)
#     img = np.array(img) / 255
#     print("step4")
#     print(img.shape)
#     h, w = img.shape[0], img.shape[1]
#     img = img.reshape((1, w, h, 3))
#     image_vector = i2v.convert(img)
#     print(image_vector.shape)
#     image_vector = image_vector.reshape(1024, 1)
#     # image_vectors = np.tile(image_vector, sample_size)
#     print(image_vector.shape)
#
#
# #
# #
# if __name__ == '__main__':
#     main()
