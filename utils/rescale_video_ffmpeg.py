import os 
import argparse 

# Rescale input video to 320p through ffmpeg

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_video_path", type=str)
    parser.add_argument("--output_video_path", type=str)
    args = parser.parse_args()

    cmd = 'ffmpeg -i {} -vf scale=320:-2 {} -hide_banner'.format(args.input_video_path, args.output_video_path)
    os.system(cmd)
