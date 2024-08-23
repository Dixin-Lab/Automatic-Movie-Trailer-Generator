from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
import time
import os
import os.path as osp 
import math
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_bar_base", type=str default='./seg_audio_shots')
    parser.add_argument("--save_audio_embs_dir", type=str, default='./audio_embs') 
    args = parser.parse_args()

    os.makedirs(args.save_audio_embs_dir, exist_ok=True) 
    audio_name = args.save_audio_base.split('/')[-1] 
    save_embs_path = osp.join(args.save_audio_embs_dir, audio_name+'.npy')

    print(f'Start encoding audio embeddings.')

    # Instantiate model
    device = 'cuda'
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device) 

    audio_files = sorted(os.listdir(args.save_bar_base), key=lambda x: int(str(x).split('.')[0]))
    audio_paths = [os.path.join(args.save_bar_base, file) for file in audio_files] 

    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)
    audio_emb = embeddings[ModalityType.AUDIO]
    np_emb = audio_emb.detach().cpu().numpy()
    np.save(save_embs_path, np_emb)

    print(f'Audio bars in {args.save_bar_base} are encoded successfully. The audio embs\'shape is {audio_emb.size()}')

