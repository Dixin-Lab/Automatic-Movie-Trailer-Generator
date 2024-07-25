# Automatic-Movie-Trailer-Generator
[ACM MM 2024 paper] An Inverse Partial Optimal Transport Framework for Music-guided Movie Trailer Generation
![scheme](img/ipot_schemes.png)

## ⏳Project Struture
```
.
├── data
│   ├── CMTD dataset
|   |   ├── audio_shot_embs (npy format, segmented audio shots)
|   |   ├── movie_shot_embs (npy format, segmented movie shots)
|   |   └── audio_movie_alignments (json format, alignment relation of audio and movie shot indices)
│   └── MV dataset
|       ├── audio_shot_embs (npy format, segmented audio shots)
|       └──  movie_shot_embs (npy format, segmented movie shots)
├── alignment
├── feature_extratction
├── segmentation
└── utils
```
## ⚙️Main Dependencies
- python=3.8.19
- pytorch=2.3.0+cu121
- numpy=1.24.1
- matplotlib=3.7.5
- scikit-learn=1.3.2
- scipy=1.10.1
- sk-video=1.1.10
- ffmpeg=1.4

Or create the environment by:
```commandline 
pip install -r requirement.txt
```

## 🎥 Dataset 
###  Dataset Download
We construct a new public comprehensive movie-trailer dataset (CMTD) for movie trailer generation and future video understanding tasks. We train and evaluate various trailer generators on this dataset. Please download the CMTD dataset from these links: [CMTD](https://drive.google.com/drive/folders/1qYKi9nsrHUiOZIAvA-uTFOsOj0rEAc26?usp=drive_link). We also provide a music video dataset (MV) for pre-training process. Please download the MV dataset from these links: [MV](https://drive.google.com/drive/folders/1FROsoTIi4lhgSHfLFJ4phE7ZFxj3udcP?usp=drive_link).

It is worth noting that due to movie copyright issues, we cannot provide the original movies. The dataset only provides the visual and acoustic features extracted by [ImageBind](https://github.com/facebookresearch/ImageBind) after we segmented the movie shot and audio shot using BaSSL.

### Movie Shot Segmentation 
We use [BaSSL](https://github.com/kakaobrain/bassl) to split each movie into movie shots and scenes, the codes can be found in ```./segmentation/scene_segmentation_bassl.py```. 
If you want to perform shot segmentation on your local video, please be aware of modifying the path for reading the video and the path for saving the segmentation results in the code.

```commandline
movie_dataset_base = '' # video data directory
movies = os.listdir(movie_dataset_base)

save_scene_dir_base = '' # save directory of scene json files 
finished_files = os.listdir(save_scene_dir_base)
```

During training phase, 

### Music Shot Segmentation 
We use [Ruptures](https://github.com/deepcharles/ruptures) to split music into movie shots and scenes, the codes can be found in ```./segmentation/scene_segmentation_bassl.py```. 
