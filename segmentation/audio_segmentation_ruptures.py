import librosa
import numpy as np
from IPython.display import Audio, display
import os 
import os.path as osp 
import argparse 
import ruptures as rpt
import json

# This code is modified based on the Ruptures code from https://github.com/deepcharles/ruptures.

# segment the input music into multiple shots based on change points through ruptures. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_audio_path", type=str)
    parser.add_argument("--save_audio_dir", type=str, default='./')

    audio = args.input_audio_path.split('/')[-1]
    audio_name = audio[:-4] # remove the ext .mp4 
    save_result_path = osp.join(args.save_scene_dir, audio_name+'.json')    # save segmentation result

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
