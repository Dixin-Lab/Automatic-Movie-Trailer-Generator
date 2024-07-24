import os 

video_dir = ''
audio_dir = ''
files = os.listdir(video_dir) 
files = sorted(files,key=lambda x: int(str(x).split('-')[0]))

for file in files: 
    # file: 1-1.mp4
    src_path = os.path.join(video_dir, file) 
    file_name = file[:-4] 
    tgt_path = os.path.join(audio_dir, file_name + '.mp3') 
    cmd = 'ffmpeg -i ' + src_path + ' -f mp3 -vn ' + tgt_path
    os.system(cmd)
    print('{} finished!!'.format(file))