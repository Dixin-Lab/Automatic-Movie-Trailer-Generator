# Automatic-Movie-Trailer-Generator
[ACMMM 2024] An Inverse Partial Optimal Transport Framework for Music-guided Movie Trailer Generation
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

Or create the environment by:
```commandline 
pip install -r requirement.txt
```

## 🎥 Dataset 
### Download dataset
We construct a new public comprehensive movie-trailer dataset (CMTD) for movie trailer generation and future video understanding tasks. We train and evaluate various trailer generators on this dataset. Please download the CMTD dataset from these links: [CMTD](https://drive.google.com/drive/folders/1qYKi9nsrHUiOZIAvA-uTFOsOj0rEAc26?usp=drive_link). We also provide a music video dataset (MV) for pre-training process. Please download the MV dataset from these links: [MV](https://drive.google.com/drive/folders/1FROsoTIi4lhgSHfLFJ4phE7ZFxj3udcP?usp=drive_link).

It is worth noting that due to movie copyright issues, we cannot provide the original movies. The dataset only provides the features extracted by [ImageBind](https://github.com/facebookresearch/ImageBind) after we segmented the movie shot and audio shot using [BaSSL](https://github.com/kakaobrain/bassl).
