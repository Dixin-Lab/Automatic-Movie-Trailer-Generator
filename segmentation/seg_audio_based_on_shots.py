import os
import os.path as osp
import math
import json
import wave
import numpy as np


def time_to_seconds(time_string):
    time_components = time_string.split(':')
    hours = int(time_components[0])
    minutes = int(time_components[1])
    seconds = int(time_components[2].split('.')[0])
    milliseconds = int(time_components[2].split('.')[1])
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def cal_audio_length(audio_info):
    dic = dict()
    for key, value in audio_info.items():
        ls = []
        for idx, time in value.items():
            ls.append(time["time"][1] - time["time"][0])
        dic[key] = ls
    return dic


def cal_movie_length(movie_info):
    ls = []
    for value in movie_info["shot_meta_list"]:
        start = time_to_seconds(value["timestamps"][0])
        end = time_to_seconds(value["timestamps"][1])
        ls.append(end - start)
    return ls


seg_json = dict()  # save the segmentation info of audio 
base = ''
save_seg_json_name = 'xxx.json'
save_bar_base = ""
scene_trailer_base = ""
audio_base = ""

files = os.listdir(audio_base)

for idx in files:
    # e.g., idx: 1-1.wav
    idx = idx[:-4]
    file_name = str(idx)
    print(file_name)

    audio_file_name = file_name + '.wav'
    audio_file_path = osp.join(audio_base, audio_file_name)

    wf = wave.open(audio_file_path, "rb")
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    duration = nframes / framerate
    print("audio file duration: %.2fs" % duration)

    movie_info_path = osp.join(scene_trailer_base, '{}.json'.format(file_name))
    with open(movie_info_path, 'r') as f:
        movie_info = json.load(f)

    movie_length_list = cal_movie_length(movie_info)

    sum = 0
    for idx in range(len(movie_length_list)):
        sum = movie_length_list[idx] + sum
        movie_length_list[idx] = sum

    print("trailer length: %.2fs" % sum)
    print('shots num: {}'.format(len(movie_length_list)))

    start_time = 0.0
    bar_idx = 0
    within_start_time = -1
    within_next_time = -1

    seg_json[file_name] = dict()
    save_bar_path = osp.join(save_bar_base, file_name)
    os.makedirs(save_bar_path, exist_ok=True)

    for shot_idx in range(len(movie_length_list)):
        next_time = movie_length_list[shot_idx]
        if next_time > duration:
            start_time = within_start_time
            next_time = within_next_time
        else:
            within_start_time = start_time
            within_next_time = next_time

        wf.setpos(int(start_time * framerate))
        wave_length = int(np.around((next_time - start_time) * framerate))

        seg_json[file_name][bar_idx] = dict()
        seg_json[file_name][bar_idx]["time"] = [start_time, next_time]

        data = wf.readframes(wave_length)

        bar_save_path = osp.join(save_bar_path, "{}.wav".format(bar_idx))
        new_wf = wave.open(bar_save_path, "wb")
        new_wf.setnchannels(nchannels)
        new_wf.setsampwidth(sampwidth)
        new_wf.setframerate(framerate)
        new_wf.writeframes(data)
        new_wf.close()

        start_time = next_time
        bar_idx += 1

    print('bars num: {}'.format(bar_idx))

with open(os.path.join(base, save_seg_json_name), 'w') as f:
    json.dump(seg_json, f)
