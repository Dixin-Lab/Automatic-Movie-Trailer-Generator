from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os
import json
import time 

# CUDA_VISIBLE_DEVICES=x python scene_segmentation_bassl.py 

movie_dataset_base = '' # video data directory
movies = os.listdir(movie_dataset_base)

save_scene_dir_base = '' # save directory of scene json files 
finished_files = os.listdir(save_scene_dir_base)

for movidx in movies: 
    print(movidx)
    movie_name = str(movidx)[:-4]

    movie_dir = os.path.join(movie_dataset_base, movie_name + '.mp4')
    saved_file_name = movie_name + '.json' 
    if saved_file_name in finished_files:
        continue
    
    video_scene_seg = pipeline(Tasks.movie_scene_segmentation, model='damo/cv_resnet50-bert_video-scene-segmentation_movienet')
    result = video_scene_seg(movie_dir)

    save_scene_dir_file = os.path.join(save_scene_dir_base, movie_name + '.json')
    with open(save_scene_dir_file, 'w') as f:
        json.dump(result, f)
