# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.nn.functional import normalize
import torch
import os
import faiss  # make faiss available

d = 1024  # dimension
base = ''  # save results in this directory 
data_dir = ''
files = os.listdir(data_dir)

for movidx in files:
    # e.g., movidx: 1-1
    testidx = movidx.split('-')[0]
    tidx = movidx.split('-')[1]

    movie_emb_path = '..../ImageBind/movie_frame_embs_test/{}.npy'.format(testidx)
    trailer_emb_path = '..../ImageBind/trailer_frame_embs_baselines/{}.npy'.format(movidx)

    xb = np.load(movie_emb_path)
    xb = normalize(torch.from_numpy(xb), p=2.0, dim=1).detach().cpu().numpy()
    nb = xb.shape[0]
    xb = np.squeeze(xb)
    xb = np.float32(xb)
    print(xb.shape)
    xq = np.load(trailer_emb_path)
    xq = normalize(torch.from_numpy(xq), p=2.0, dim=1).detach().cpu().numpy()
    nq = xq.shape[0]
    xq = np.squeeze(xq)
    xq = np.float32(xq)
    print(xq.shape)

    res = faiss.StandardGpuResources()  # use a single GPU
    # Using a flat index
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(xb)  # add vectors to the index
    print(gpu_index_flat.ntotal)

    k = 4  # we want to see 4 nearest neighbors
    D, I = gpu_index_flat.search(xq, k)  # actual search

    np.save(os.path.join(base, 'result-flat-{}.npy'.format(movidx)), I)
