from pytube import YouTube
import moviepy.editor as mp
from PIL import Image
# import imageio

video_id = "h6yJEHHT5eA"
try:
    youtube = YouTube("https://www.youtube.com/watch?v="+video_id)
    youtube.set_filename('tmp')
except:
    print("there is no video")

try:
    video = youtube.get('mp4', '360p')
except:
    print("there is no video for this setting")

video.download(".")

clip = mp.VideoFileClip("tmp.mp4")
clip.audio.write_audiofile("tmp.wav")

fps = clip.fps

num_frames = int(fps * clip.duration)

# to get frame
clip.get_frame(0)
# to get image instance from numpy array
img = Image.fromarray(clip.get_frame(0))
img.thumbnail([30,30], Image.ANTIALIAS)
# reader = imageio.get_reader('tmp.mp4')