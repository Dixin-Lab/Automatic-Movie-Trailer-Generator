import os
import os.path as osp

savebase = ''

movies_dir = ''  # the movies directory 
movies = os.listdir(movies_dir)
movies = sorted(movies, key=lambda x: int(str(x).split('-')[0]))

have_dealed = os.listdir(dir1)

for movid in movies:
    # movid: 1-1.mp4
    movid = movid[:-4]  # 1-1
    if movid in have_dealed:
        continue
    print(movid)

    movie_path = osp.join(movies_dir, str(movid) + '.mp4')
    save_dir = osp.join(savebase, str(movid))
    os.makedirs(save_dir, exist_ok=True)
    cmd = 'ffmpeg -i ' + movie_path + ' ' + save_dir + '/%d.png'
    os.system(cmd)
