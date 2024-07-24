import os
import os.path as osp
import json
import numpy as np

shot_align_base = ''
bar_shot_align_base = ''
scene_trailer_base = ''

# the audio shot segmentation is based on the timestamps of video's shot segmentation
# e.g., the number of audio shots is the same as the number of video shots
bar_seg_info_path = ''
with open(bar_seg_info_path, 'r') as f:
    seg_info = json.load(f)


def tranform_timestamp_to_seconds(timestamp):
    # e.g. timestamp: 00:00:00.000
    hour, minute, second = timestamp.split(':')
    second_all = float(hour) * 3600.0 + float(minute) * 60.0 + float(second)
    return second_all


files = os.listdir(shot_align_base)
for file in files:
    # e.g., 1-1.json
    movidx = file.split('.')[0]
    bar_info = seg_info[movidx]
    n_bar = len(bar_info.keys())

    # read in trailer's scene_seg json 
    t_scene_json_path = osp.join(scene_trailer_base, movidx + '.json')
    with open(t_scene_json_path, 'r') as f:
        t_scene_json = json.load(f)["shot_meta_list"]

    shot_align_json_path = osp.join(shot_align_base, movidx + '.json')
    with open(shot_align_json_path, 'r') as f:
        shot_align_json = json.load(f)

    n_shot = len(t_scene_json)

    # define save file of the alignment between bar and shots 
    save_file_path = osp.join(bar_shot_align_base, movidx + '.json')
    save_json = dict()

    for baridx in range(n_bar):
        st = bar_info[str(baridx)]["time"][0]
        et = bar_info[str(baridx)]["time"][1]
        # print('st: {}  et: {}'.format(st, et))
        save_list = list()
        shotidx = baridx
        movie_shot_idx = shot_align_json[str(shotidx)]
        save_list.append([shotidx, movie_shot_idx])
        save_json[baridx] = save_list

    with open(save_file_path, 'w') as f:
        json.dump(save_json, f)
