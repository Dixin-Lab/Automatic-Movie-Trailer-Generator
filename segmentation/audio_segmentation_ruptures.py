import librosa
import numpy as np
from IPython.display import Audio, display
import os 
import os.path as osp 
import argparse 
import ruptures as rpt
import json
import math
import wave 

# This code is modified based on the Ruptures code from https://github.com/deepcharles/ruptures.

# segment the input music into multiple shots based on change points through ruptures. 

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_audio_path", type=str)
    parser.add_argument("--save_scene_dir", type=str, default='./')
    parser.add_argument("--save_bar_dir", type=str, default='./seg_audio_shots')

    os.makedirs(args.save_scene_dir, exist_ok=True)
    os.makedirs(args.save_bar_dir, exist_ok=True)

    audio = args.input_audio_path.split('/')[-1]
    audio_name = audio[:-4] # remove the ext .wav 
    save_result_path = osp.join(args.save_scene_dir, audio_name+'.json')    # save segmentation result

    save_bar_base = osp.join(args.save_bar_dir, audio_name) 
    os.makedirs(save_bar_base, exist_ok=True)

    print(f'Start processing audio {args.input_audio_path}.')

    signal, sampling_rate = librosa.load(args.input_audio_path)

    # Compute the onset strength
    hop_length_tempo = 256
    oenv = librosa.onset.onset_strength(y=signal, sr=sampling_rate, hop_length=hop_length_tempo)

    # Compute the tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv,
        sr=sampling_rate,
        hop_length=hop_length_tempo,
    )

    # choose the segmentation model
    algo = rpt.KernelCPD(kernel="linear").fit(tempogram.T)

    # Segmentation
    duration = librosa.get_duration(filename=audio_path)
    n_bkps = int(duration / 2) - 1

    bkps = algo.predict(n_bkps=n_bkps)
    # Convert the estimated change points (frame counts) to actual timestamps
    bkps_times = librosa.frames_to_time(bkps, sr=sampling_rate, hop_length=hop_length_tempo)
        
    bkps_times_list = bkps_times.tolist()
    #print("audio segmentation is : {}".format(index, bkps_times_list))

    with open(save_result_path, 'w') as f:
        json.dump(bkps_times_list, f)
    
    # based on the bkps_times_list, segment the raw music into music shots. 
    wf = wave.open(args.input_audio_path, "rb")
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    duration = nframes / framerate 

    pre_time = 0.0
    for shot_idx in range(len(bkps_times_list)): 
        now_time = bkps_times_list[shot_idx] 

        wf.setpos(int(pre_time * framerate))
        wave_length = int(np.around((now_time - pre_time) * framerate))
        data = wf.readframes(wave_length)
        bar_save_path = osp.join(save_bar_base, "{}.wav".format(shot_idx))
        new_wf = wave.open(bar_save_path, "wb")
        new_wf.setnchannels(nchannels)
        new_wf.setsampwidth(sampwidth)
        new_wf.setframerate(framerate)
        new_wf.writeframes(data)
        new_wf.close()

        pre_time = now_time

    print(f'Input audio {args.input_audio_path} is segmented successfully. Audio shots are saved in {save_bar_base}, {len(bkps_times_list)} bars in total.')
