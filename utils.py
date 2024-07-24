import torch
import numpy as np
import os
import json
import cv2
import wave
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skvideo.io
import ffmpeg
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import os.path as osp


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_best_model(args, epoch, best_metric, model, optimizer, checkpoint=False):
    print("Saving model to {} with training loss = {}".format(args.save_path, best_metric))
    if checkpoint and epoch == args.epochs - 1:
        save_dict = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        file_path = os.path.join(args.save_path,
                                 "structure={}_epochs={}_lr={}_w_at={}_w_am={}_w_tm={}_dropout={}_tem={}_onlyaudio_mse_1v1".format(
                                     args.structure, args.epochs, args.lr, args.weight_at, args.weight_am,
                                     args.weight_tm, args.dropout, args.temperature))
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        torch.save(save_dict, os.path.join(file_path, "{}.pt".format(epoch + 1)))


def check_best_score(epoch, best_score, val_loss, model, optimizer, args):
    if not best_score:
        save_best_model(args, epoch, val_loss, model, optimizer)
        return val_loss
    if val_loss > best_score:
        best_score = val_loss
        save_best_model(args, epoch, best_score, model, optimizer)
    return best_score


def time_to_seconds(time_string):
    time_components = time_string.split(':')

    hours = int(time_components[0])
    minutes = int(time_components[1])
    seconds = int(time_components[2].split('.')[0])
    milliseconds = int(time_components[2].split('.')[1])

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def seconds_to_time(seconds):
    # input: float:  seconds
    # return: time_string hh:mm:ss.xxx
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return "{:02d}:{:02d}:{:02d}.{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))


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
        start = time_to_seconds(value["timestamps"][0][0])
        end = time_to_seconds(value["timestamps"][1][0])
        ls.append(end - start)
    return ls


def cal_audio_length_fortest(audio_info):
    ls = []
    for key, value in audio_info.items():
        for idx, time in value.items():
            ls.append(time[1] - time[0])
    return ls


def cal_audio_length_rup(audio_info):
    # input is a list of second timestamp 
    ls = []
    last = 0.
    for item in audio_info:
        ls.append(item - last)
        last = item
    return ls


def cal_movie_length_fortest(movie_info):
    ls = []
    for value in movie_info["shot_meta_list"]:
        start = time_to_seconds(value["timestamps"][0])
        end = time_to_seconds(value["timestamps"][1])
        ls.append(end - start)
    return ls


def get_duration_from_ffmpeg(filename):
    probe = ffmpeg.probe(filename)
    format = probe['format']
    duration = format['duration']
    size = int(format['size']) / 1024 / 1024
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    fps = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    return float(duration), int(num_frames)


def seg_video_by_timestamp(movidx, j, l_t, duration_t, save_seg_shots_base2):
    movie_base = '..../test_movies_intra'
    src_path = osp.join(movie_base, movidx + '.mp4')

    tgt_path = osp.join(save_seg_shots_base2, str(j) + '.mp4')
    cmd = 'ffmpeg -i {} -ss {} -t {} -c copy {} -y'.format(src_path, l_t, duration_t, tgt_path)
    # cmd = 'ffmpeg -y -i {} -ss {} -t {} -c:a aac -c:v libx264 {}'.format(src_path,l_t,duration_t,tgt_path)
    os.system(cmd)
    print('movie {}: {}-th shot segmented done.'.format(movidx, str(j)))


def video_seg_concate(movidx, alignment, sig_score):
    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    # 3. add fade in and fade out for each shot 
    # 4. concate all video shots. 

    # # Input: (Suppose I moive shots, J trailer audio shots)
    # alignment: [J,1]: [a_i -> m_j] 
    # sig_score: [I,1] 
    # movidx: \in [1,2,...,8]

    movidx = str(movidx)

    # save the segmented shots based on audio
    save_seg_shots_base = '..../test_seg_shots'
    # save the segmented shots based on audio (fade in and fade out)
    save_seg_shots_fade_base = '..../test_seg_shots_fade'
    output_video_base = '..../output-video'
    save_seg_shots_base2 = osp.join(save_seg_shots_base, movidx)
    os.makedirs(save_seg_shots_base2, exist_ok=True)

    scene_movie_base = '..../scene_test_movies'
    scene_movie_info_path = osp.join(scene_movie_base, movidx + '.json')
    with open(scene_movie_info_path, 'r') as f:
        scene_movie_info = json.load(f)

    bar_seg_info_path = '..../ruptures_audio_segmentation_test_MT_2s.json'
    with open(bar_seg_info_path, 'r') as f:
        bar_seg_info = json.load(f)

    # calculate each shot's duration, each bar's duration 
    shot_length = cal_movie_length_fortest(scene_movie_info)
    audio_length = cal_audio_length_rup(bar_seg_info[movidx])

    # 1. Depend on the one-to-one alignment, find the right corresponding timestamps for each audio shot,
    # 2. segment raw movie based on timestamps, derive multiple video shots 
    J = len(audio_length)
    I = len(shot_length)

    for j in range(J):
        i = alignment[j]
        if shot_length[i] >= audio_length[j]:
            # expand from the mid time of the shot 
            t0 = scene_movie_info["shot_meta_list"][i]["timestamps"][0]
            t1 = scene_movie_info["shot_meta_list"][i]["timestamps"][1]
            t0_seconds = time_to_seconds(t0)
            t1_seconds = time_to_seconds(t1)
            mid_seconds = (t0_seconds + t1_seconds) / 2.0
            l_seconds = mid_seconds - audio_length[j] / 2.0
            # r_seconds = mid_seconds + audio_length[j]/2.0
            l_t = seconds_to_time(l_seconds)
            duration = audio_length[j]
            duration_t = seconds_to_time(duration)
            seg_video_by_timestamp(movidx, j, l_t, duration_t, save_seg_shots_base2)
        else:
            difference = audio_length[j] - shot_length[i]
            l = i
            r = i
            t0 = scene_movie_info["shot_meta_list"][i]["timestamps"][0]
            t1 = scene_movie_info["shot_meta_list"][i]["timestamps"][1]
            t0_seconds = time_to_seconds(t0)
            t1_seconds = time_to_seconds(t1)
            left_seconds = t0_seconds
            while difference:
                if l > 0 and r < I - 1:
                    if sig_score[l - 1] >= sig_score[r + 1]:
                        if shot_length[l - 1] >= difference:
                            t_r = scene_movie_info["shot_meta_list"][l - 1]["timestamps"][1]
                            t_r_seconds = time_to_seconds(t_r)
                            left_seconds = t_r_seconds - difference
                            difference = 0
                        else:
                            difference -= shot_length[l - 1]
                        l = l - 1
                    else:
                        if shot_length[r + 1] >= difference:
                            difference = 0
                        else:
                            difference -= shot_length[r + 1]
                        r = r + 1
                elif r < I - 1:
                    if shot_length[r + 1] >= difference:
                        difference = 0
                    else:
                        difference -= shot_length[r + 1]
                    r = r + 1
                elif l > 0:
                    t_r = scene_movie_info["shot_meta_list"][l - 1]["timestamps"][1]
                    t_r_seconds = time_to_seconds(t_r)
                    left_seconds = t_r_seconds - difference
                    difference = 0
                    l = l - 1
            # now: seg from left_seconds, duration = audio_length[j]
            l_t = seconds_to_time(left_seconds)
            duration = audio_length[j]
            duration_t = seconds_to_time(duration)
            seg_video_by_timestamp(movidx, j, l_t, duration_t, save_seg_shots_base2)

    print('begin deal with fade in and fade out.')

    # 3. add fade in and fade out for each shot 
    output_file = 'output-test-' + str(movidx) + '-3s-fade.mp4'
    tmp_video_file = 'output-test-' + str(movidx) + '-3s-fade-silent.mp4'
    shot_base = os.path.join(save_seg_shots_base, str(movidx))
    shot_idx_list = range(J)
    tmp_txt_file_path = '/home/yutong2/workspace/Sidan/tmp{}.txt'.format(movidx)

    # # make each shot in shot_idx_list in style of fade in and out
    shot_tmp_base = os.path.join(save_seg_shots_fade_base, str(movidx))
    os.makedirs(shot_tmp_base, exist_ok=True)
    for shot_idx in shot_idx_list:
        shot_file_name = "{}.mp4".format(shot_idx)
        path_in = os.path.join(shot_base, shot_file_name)
        path_out = os.path.join(shot_tmp_base, shot_file_name)
        duration, nframe = get_duration_from_ffmpeg(path_in)
        if nframe > 60:  # 
            cmd = 'ffmpeg -i "{}" -vf "fade=in:0:10,fade=out:{}:8" "{}" -y'.format(path_in, int(nframe - 8), path_out)
        else:
            cmd = 'cp {} {}'.format(path_in, path_out)
        os.system(cmd)

    # 4. concate all video shots. 
    # write the required txt file 

    print('begin to concate all video shots.')

    with open(tmp_txt_file_path, 'w') as f:
        for shot_idx in shot_idx_list:
            shot_file = "{}.mp4".format(shot_idx)
            shot_file_path = os.path.join(shot_tmp_base, shot_file)
            write_line = "file '{}'\n".format(shot_file_path)
            f.writelines(write_line)

    output_file_path = os.path.join(output_video_base, output_file)
    tmp_video_file_path = os.path.join(output_video_base, tmp_video_file)

    # generate initial trailer based on the movie shots concatenation
    cmd1 = 'ffmpeg -f concat -safe 0 -i {} -c copy {} -y'.format(tmp_txt_file_path, output_file_path)
    os.system(cmd1)

    # remove the audio in the generated trailer (optional)
    cmd2 = 'ffmpeg -i {} -c:v copy -an {} -y'.format(output_file_path, tmp_video_file_path)
    os.system(cmd2)
