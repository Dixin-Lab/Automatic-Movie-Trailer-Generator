import matplotlib.pyplot as plt
import librosa
import numpy as np
from IPython.display import Audio, display
import os
import ruptures as rpt
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = x

# This code is modified based on the Ruptures code from https://github.com/deepcharles/ruptures.

def fig_ax(figsize=(30, 5), dpi=150):
    """Return a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)

audio_file_path = ''  # music data path
save_result_base = ''  # save segmentation result

audio_path = os.path.join(audio_file_path)
signal, sampling_rate = librosa.load(audio_path)

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

# Displaying results
fig, ax = fig_ax()
_ = librosa.display.specshow(
    tempogram,
    ax=ax,
    x_axis="s",
    y_axis="tempo",
    hop_length=hop_length_tempo,
    sr=sampling_rate,
)
    
bkps_times_list = bkps_times.tolist()
print("audio segmentation is : {}".format(index, bkps_times_list))

with open(save_result_base, 'w') as f:
    json.dump(bkps_times_list, f)
