import os 

# Implement intra-frame coding of the video so that the duration error in concatenating the video shots is as small as possible.

src_video_dir = '' 
tgt_video_dir = ''
files = os.listdir(src_video_dir)

for file in files: 
    src_path = os.path.join(src_video_dir, file)
    tgt_path = os.path.join(tgt_video_dir, file)
    cmd = 'ffmpeg -i {} -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -x264-params "keyint=1:min-keyint=1:scenecut=0" -c:a copy {}'.format(src_path, tgt_path)
    os.system(cmd)
