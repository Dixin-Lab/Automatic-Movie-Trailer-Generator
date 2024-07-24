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
# Load data

base = ''  # data directory that contains audio shots 
movfiles = sorted(os.listdir(base), key=lambda x: int(str(x)))

save_emb_base = ""

for idx in movfiles:
    # e.g., idx: 1-1
    movidx = str(idx)
    print(movidx)

    audio_base = os.path.join(base, '{}'.format(movidx))
    audio_files = sorted(os.listdir(audio_base), key=lambda x: int(str(x).split('.')[0]))
    audio_paths = [os.path.join(audio_base, file) for file in audio_files]

    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    audio_emb = embeddings[ModalityType.AUDIO]
    print(audio_emb.size())
    np_emb = audio_emb.detach().cpu().numpy()
    save_path = '{}.npy'.format(movidx)
    np.save(os.path.join(save_emb_base, save_path), np_emb)
