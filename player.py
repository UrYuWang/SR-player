# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import DCSCN
from helper import args
from moviepy.video.io.VideoFileClip import VideoFileClip
import pygame

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
FLAGS = args.get()


def main(_):
    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

    video=FLAGS.file
    lrclip=VideoFileClip(video).subclip(0,20)
    audioclip=lrclip.audio
    audioclip.write_audiofile('audio'+video[:-4]+'.mp3')
    hrclip=lrclip.fl_image(model.doframe)
    # hrclip.ipython_display()
    # lrclip.ipython_display()
    '''
        Uncomment the line if you run the code in jupyter notebook
    '''
    hroutput='hr'+video
    hrclip.write_videofile(hroutput, audio='audio'+video[:-4]+'.mp3', threads=8, progress_bar=False)
    # hrclip.write_videofile(hroutput, audio=False)

if __name__ == '__main__':
    tf.app.run()
