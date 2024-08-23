from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
import time
import os
import math
import os.path as osp 
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_shot_base", type=str default='./results/shot_split_video')
    parser.add_argument("--save_video_embs_dir", type=str, default='./video_embs') 
    args = parser.parse_args()

    os.makedirs(args.save_video_embs_dir, exist_ok=True) 
    video_name = args.save_shot_base.split('/')[-1] 
    save_embs_path = osp.join(args.save_video_embs_dir, video_name+'.npy')

    print(f'Start encoding video embeddings.')

    # Instantiate model
    device = 'cuda'
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device) 

    video_files = sorted(os.listdir(video_base_dir), key=lambda x: int(str(x)[:-4].split('_')[-1]))
    video_paths = [os.path.join(args.save_shot_base, file) for file in video_files] 

    shot_n = len(video_paths)
    shot_subset_len = 1
    shot_subset_n = math.ceil(len(video_paths) / shot_subset_len)

    for subset_idx in range(shot_subset_n):
        shot_subset_l = subset_idx * shot_subset_len
        shot_subset_r = min((subset_idx + 1) * shot_subset_len, shot_n)
        shot_subset_paths = video_paths[shot_subset_l:shot_subset_r]
        # print('subset idx : {}'.format(subset_idx))
        # print('subset len : {}'.format(shot_subset_r - shot_subset_l))

        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(shot_subset_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        vision_emb = embeddings[ModalityType.VISION]
        np_v_emb = vision_emb.detach().cpu().numpy()
        v_save_file = 'movie_shot_embs_{}_{}.npy'.format(video_name, subset_idx)
        np.save(v_save_file, np_v_emb) 
    
    temp = np.zeros((1,1024))
    for subset_idx in range(shot_subset_n):
        v_save_path = 'movie_shot_embs_{}_{}.npy'.format(video_name, subset_idx)
        sdata = np.load(v_save_path, allow_pickle=True)
        temp = np.concatenate((temp, sdata), axis=0)

    np.save(os.path.join(args.save_video_embs_dir, video_name+'.npy'), temp[1:])

    for subset_idx in range(shot_subset_n):
        v_save_path = 'movie_shot_embs_{}_{}.npy'.format(video_name, subset_idx)
        os.system('rm {}'.format(v_save_path))

    print(f'Video are encoded successfully. The video embs\'shape is {temp[1:].shape}. The video embs are saved in {args.save_video_embs_dir}')
