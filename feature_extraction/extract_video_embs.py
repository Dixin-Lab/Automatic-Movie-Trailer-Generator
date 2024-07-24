from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
import time
import os
import math

device = "cuda:5" if torch.cuda.is_available() else "cpu"
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

data_base = ''
files = os.listdir(data_base)

for movid in files:
    # e.g., movid: 1-1
    print(movid)

    video_base_dir = '..../trailer-scenes/results/shot_split_video/{}'.format(movid)
    video_paths = sorted(os.listdir(video_base_dir), key=lambda x: int(str(x)[:-4].split('_')[-1]))

    shot_n = len(video_paths)
    shot_subset_len = 1
    shot_subset_n = math.ceil(len(video_paths) / shot_subset_len)

    for idx in range(len(video_paths)):
        video_paths[idx] = os.path.join(video_base_dir, video_paths[idx])

    for subset_idx in range(shot_subset_n):
        shot_subset_l = subset_idx * shot_subset_len
        shot_subset_r = min((subset_idx + 1) * shot_subset_len, shot_n)
        shot_subset_paths = video_paths[shot_subset_l:shot_subset_r]
        print('subset idx : {}'.format(subset_idx))
        print('subset len : {}'.format(shot_subset_r - shot_subset_l))

        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(shot_subset_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        vision_emb = embeddings[ModalityType.VISION]
        print(vision_emb.size())

        np_v_emb = vision_emb.detach().cpu().numpy()
        v_save_path = 'trailer_shot_embs_{}_{}.npy'.format(movid, subset_idx)
        np.save(v_save_path, np_v_emb)

    rst_save_base = '..../trailer_shot_embs'
    v_save_path_final = str(movid) + '.npy'
    temp = []

    for subset_idx in range(shot_subset_n):
        v_save_path = 'trailer_shot_embs_{}_{}.npy'.format(movid, subset_idx)
        sdata = np.load(v_save_path, allow_pickle=True)
        temp.append(sdata)

    np.save(os.path.join(rst_save_base, v_save_path_final), temp)

    import os

    for subset_idx in range(shot_subset_n):
        v_save_path = 'trailer_shot_embs_{}_{}.npy'.format(movid, subset_idx)
        os.system('rm {}'.format(v_save_path))
