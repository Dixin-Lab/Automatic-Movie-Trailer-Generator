import os 
import os.path as osp 
import argparse 
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import json
import time 

# segment the input long video into multiple shots based on the actions and scene changes. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_video_path", type=str)
    parser.add_argument("--save_scene_dir", type=str, default='./')
    args = parser.parse_args()

    video = args.input_video_path.split('/')[-1]
    video_name = video[:-4] # remove the ext .mp4 
    save_scene_path = osp.join(args.save_scene_dir, video_name+'.json')

    video_scene_seg = pipeline(Tasks.movie_scene_segmentation, model='damo/cv_resnet50-bert_video-scene-segmentation_movienet')
    result = video_scene_seg(args.input_video_path) 

    # save the scene split json file.
    with open(save_scene_path, 'w') as f:
        json.dump(result, f)
