# A TensorFlow implementation of DeepMind's WaveNet paper

[![Build Status](https://travis-ci.org/ibab/tensorflow-wavenet.svg?branch=master)](https://travis-ci.org/ibab/tensorflow-wavenet)

This is a TensorFlow implementation of the [WaveNet generative neural
network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) for audio generation.

<table style="border-collapse: collapse">
<tr>
<td>
<p>
The WaveNet neural network architecture directly generates a raw audio waveform,
showing excellent results in text-to-speech and general audio generation (see the
DeepMind blog post and paper for details).
</p>
<p>
The network models the conditional probability to generate the next
sample in the audio waveform, given all previous samples and possibly
additional parameters.
</p>
<p>
After an audio preprocessing step, the input waveform is quantized to a fixed integer range.
The integer amplitudes are then one-hot encoded to produce a tensor of shape <code>(num_samples, num_channels)</code>.
</p>
<p>
A convolutional layer that only accesses the current and previous inputs then reduces the channel dimension.
</p>
<p>
The core of the network is constructed as a stack of <em>causal dilated layers</em>, each of which is a
dilated convolution (convolution with holes), which only accesses the current and past audio samples.
</p>
<p>
The outputs of all layers are combined and extended back to the original number
of channels by a series of dense postprocessing layers, followed by a softmax
function to transform the outputs into a categorical distribution.
</p>
<p>
The loss function is the cross-entropy between the output for each timestep and the input at the next timestep.
</p>
<p>
In this repository, the network implementation can be found in <a href="./wavenet/model.py">model.py</a>.
</p>
</td>
<td width="300">
<img src="images/network.png" width="300"></img>
</td>
</tr>
</table>

## Requirements

TensorFlow needs to be installed before running the training script.
Code is tested on TensorFlow version 1.0.1 for Python 2.7 and Python 3.5.

In addition, [librosa](https://github.com/librosa/librosa) must be installed for reading and writing audio.

To install the required python packages, run
```bash
pip install -r requirements.txt
```

For GPU support, use
```bash
pip install -r requirements_gpu.txt
```
To use moviepy library, you may need to install ffmpeg.
```bash
sudo apt install ffmpeg
```
or/and may need to execute
```
imageio.plugins.ffmpeg.download()
```
in python

## Training the network

The training data is already prepared in the ./data directory. However, in order to use vgg19 networks, the .npy file has to be downloaded form [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).

In order to train the network, execute
```bash
python train.py --data_dir=corpus
```
to train the network, where `corpus` is a directory containing `.wav` files.
The script will recursively collect all `.wav` files in the directory.

You can see documentation on each of the training settings by running
```bash
python train.py --help
```

You can find the configuration of the model parameters in [`wavenet_params.json`](./wavenet_params.json).
These need to stay the same between training and generation.

### Global Conditioning
Global conditioning refers to modifying the model such that the id of a set of mutually-exclusive categories is specified during training and generation of .wav file.
In the case of the VCTK, this id is the integer id of the speaker, of which there are over a hundred.
This allows (indeed requires) that a speaker id be specified at time of generation to select which of the speakers it should mimic. For more details see the paper or source code.

### Training with Local Conditioning
The instructions above for training refer to training without local conditioning. To train with local conditioning, specify command-line arguments as follows:
```
python train.py --lc_channels=512 --isDebug=False --num_steps=1000 --generate_every=5
```
As the vector describes an input image has 512 dimension, --lc_channels is 512. Therefore, if you change the size of the input image, you need to change it.

--isDebug tells the train.py script if you are debugging or not.

--num_steps is the number of epochs to train the model. Defalt is 1000.

--generate_every is the number of epochs to calculate validation score and generate sound file after. Defalt is 5.

The --lc_channels argument does two things:
* It tells the train.py script that
it should build a model that includes global conditioning.
* It specifies the
size of the embedding vector that is looked up based on the id of the speaker.

The global conditioning logic in train.py and audio_reader.py is "hard-wired" to the VCTK corpus at the moment in that it expects to be able to determine the speaker id from the pattern of file naming used in VCTK, but can be easily be modified.

## Generating audio

[Example output](https://soundcloud.com/user-731806733/tensorflow-wavenet-500-msec-88k-train-steps)
generated by @jyegerlehner based on speaker 280 from the VCTK corpus.

You can use the `generate.py` script to generate audio using a previously trained model.

### Generating without Global Conditioning
Run
```
python generate.py --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```
where `logdir/train/2017-02-13T16-45-34/model.ckpt-80000` needs to be a path to previously saved model (without extension).
The `--samples` parameter specifies how many audio samples you would like to generate (16000 corresponds to 1 second by default).

The generated waveform can be played back using TensorBoard, or stored as a
`.wav` file by using the `--wav_out_path` parameter:
```
python generate.py --wav_out_path=generated.wav --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```

Passing `--save_every` in addition to `--wav_out_path` will save the in-progress wav file every n samples.
```
python generate.py --wav_out_path=generated.wav --save_every 2000 --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```

Fast generation is enabled by default.
It uses the implementation from the [Fast Wavenet](https://github.com/tomlepaine/fast-wavenet) repository.
You can follow the link for an explanation of how it works.
This reduces the time needed to generate samples to a few minutes.

To disable fast generation:
```
python generate.py --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000 --fast_generation=false
```

### Generating with Global Conditioning
Generate from a model incorporating global conditioning as follows:
```
python generate.py --samples 16000  --wav_out_path speaker311.wav --gc_channels=32 --gc_cardinality=377 --gc_id=311 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```
Where:

`--gc_channels=32` specifies 32 is the size of the embedding vector, and
must match what was specified when training.

`--gc_cardinality=377` is required
as 376 is the largest id of a speaker in the VCTK corpus. If some other corpus
is used, then this number should match what is automatically determined and
printed out by the train.py script at training time.

`--gc_id=311` specifies the id of speaker, speaker 311, for which a sample is
to be generated.

## Running tests

Install the test requirements
```
pip install -r requirements_test.txt
```

Run the test suite
```
./ci/test.sh
```

## Missing features

Currently there is no local conditioning on extra information which would allow
context stacks or controlling what speech is generated.


## Related projects

- [tex-wavenet](https://github.com/Zeta36/tensorflow-tex-wavenet), a WaveNet for text generation.
- [image-wavenet](https://github.com/Zeta36/tensorflow-image-wavenet), a WaveNet for image generation.
