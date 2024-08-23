import os 
import argparse 

# Implement intra-frame coding of the video so that the duration error in concatenating the video shots is as small as possible.

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_video_path", type=str)
    parser.add_argument("--output_video_path", type=str)

    cmd = 'ffmpeg -i {} -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -x264-params "keyint=1:min-keyint=1:scenecut=0" -c:a copy {}'.format(args.input_video_path, args.output_video_path)
    os.system(cmd)
