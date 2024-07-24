import os 

base = ''
tgt_base = ''
movies = os.listdir(base) 

for movie in movies: 
    src_path = os.path.join(base, movie) 
    tgt_path = os.path.join(tgt_base, movie) 
    cmd = 'ffmpeg -i ' + src_path + ' -vf scale=320:-2 ' + tgt_path + ' -hide_banner'
    os.system(cmd)
