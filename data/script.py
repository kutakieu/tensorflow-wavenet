# python code to extract sound data from the videos and save as .wav files

import librosa
import moviepy.editor as mp
from os import listdir
from os.path import isfile, join

path = "./"

filenames = [f for f in listdir(path) if isfile(join(path, f))]

for file_name in filenames:
    print(file_name)
    if file_name.split(".")[1] == "mp4":
        clip = mp.VideoFileClip(path + file_name)

        clip.audio.write_audiofile(path + file_name.split(".")[0] + ".wav")
        