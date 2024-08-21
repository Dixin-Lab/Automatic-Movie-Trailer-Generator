import torch
import numpy as np
import os
import json
import cv2
import wave
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skvideo.io
import ffmpeg
import torch.nn.functional as F
from model import VA_encoder_self_cross_sigmoid
import os.path as osp
import argparse
import torch.optim as optim
import random
from torch.nn.functional import normalize
from sinkhorn import SinkhornDistance
from scipy.optimize import linear_sum_assignment
import ot
from utils import video_seg_concate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='5')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--delta', default=1.0, type=float, help='relaxtion parameter')
    parser.add_argument('--lamda', default=1.0, type=float, help='parameter controlling the regularity of distance for pi')
    parser.add_argument('--eta', default=1.0, type=float, help='weight of duration matrix')
    parser.add_argument("--test_trailer_audio_base", type=str, default='..../CMTD/test_audio_shot_embs')
    parser.add_argument("--test_movie_shot_base", type=str, default='..../CMTD/test_movie_shot_embs')
    parser.add_argument("--movie_shot_info_path", type=str, default='..../CMTD/scene_test_movies')
    parser.add_argument("--audio_bar_info_path", type=str, default='..../CMTD/ruptures_audio_segmentation.json')
    parser.add_argument("--model_path", type=str, default='..../network_500.net')
    args = parser.parse_args()
    return args


def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split('.')
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C


def min_distance_assignment(distances):
    row_ind, col_ind = linear_sum_assignment(distances)
    total_distance = distances[row_ind, col_ind].sum()
    return total_distance, list(zip(row_ind, col_ind))


def normalize_list(lst):
    total_sum = sum(lst)
    return [x / total_sum for x in lst]


def trailer_generator(video_num):
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    model_path = args.model_path
    test_movie_shot_base = args.test_movie_shot_embs
    test_trailer_audio_base_rups = args.test_audio_shot_embs

    model = VA_encoder_self_cross_sigmoid(input_dim=args.input_size, model_dim=args.hidden_size).to(device)
    model.load_state_dict(torch.load(net_path))
    model.eval()

    test_video_file = "{}.npy".format(video_num)

    with torch.no_grad():

        m_shot_emb = torch.Tensor(np.load(osp.join(test_movie_shot_base, test_video_file))).to(device)
        # set a initial range for the movie shots: 
        # e.g, [2%, 90%]
        shot_l = int(m_shot_emb.size(0)*0.02)
        shot_r = int(m_shot_emb.size(0)*0.90)
        new_movie_shots_index = torch.arange(shot_l, shot_r).to(device)

        # check mu based on rup RTS features
        t_bar_emb_rups = torch.Tensor(np.load(osp.join(test_trailer_audio_base_rups, test_video_file))).to(device)
        # print("rups number: {}".format(t_bar_emb_rups.size(0)))
        # print('m shot emb size: {}'.format(m_shot_emb.size()))

        # normalize embeddings
        m_shot_emb = normalize(m_shot_emb, p=2.0, dim=1)
        t_bar_emb_rups = normalize(t_bar_emb_rups, p=2.0, dim=1)

        m_shot_emb, mu, t_bar_emb_rups = model(m_shot_emb, t_bar_emb_rups)

        # normalize embeddings
        m_shot_emb = normalize(m_shot_emb, p=2.0, dim=1)
        t_bar_emb_rups = normalize(t_bar_emb_rups, p=2.0, dim=1)

        # calculate duration matrix
        new_mu = mu[new_movie_shots_index]
        mu_top_values, mu_top_indices = torch.topk(new_mu.view(-1), k=t_bar_emb_rups.size(0), largest=True)
        new_mu_top_indices = new_movie_shots_index[mu_top_indices]
        mu_np = mu.view(-1).detach().cpu().numpy()

        m_shot_choose = m_shot_emb[new_mu_top_indices]
        choose_movie_shot_duration = []
        movie_shot_info_path = args.movie_shot_info_path
        with open(os.path.join(movie_shot_info_path, "{}.json".format(video_num)),'r') as f:
            movie_shot_info = json.load(f)
        movie_shot_duration_info = movie_shot_info["shot_meta_list"]
        choose_movie_shot_duration_info = [movie_shot_duration_info[index] for index in mu_top_indices]
        for value in choose_movie_shot_duration_info:
            choose_movie_shot_duration.append(time_to_seconds(value["timestamps"][1])-time_to_seconds(value["timestamps"][0]))
        
        rup_audio_bar_duration = []
        audio_bar_info_path = args.audio_bar_info_path
        with open(audio_bar_info_path,'r') as f:
            audio_bar_info = json.load(f)
        current_audio_bar_info = audio_bar_info["{}".format(video_num)]
        for i in range(len(current_audio_bar_info)):
            if i == 0:
                rup_audio_bar_duration.append(current_audio_bar_info[i])
            else:
                rup_audio_bar_duration.append(current_audio_bar_info[i]-current_audio_bar_info[i-1])

        duration_matrix = torch.zeros(m_shot_choose.size(0), t_bar_emb_rups.size(0))
        for i in range(m_shot_choose.size(0)):
            for j in range(t_bar_emb_rups.size(0)):
                duration_matrix[i, j] = abs(choose_movie_shot_duration[i] - rup_audio_bar_duration[j])
        duration_matrix = duration_matrix.to(device)

        # calculate distance matrix
        distance_matrix = cost_matrix(m_shot_choose, t_bar_emb_rups).to(device)

        # final matrix
        duration_matrix_np = duration_matrix.detach().cpu().numpy()
        distance_matrix_np = distance_matrix.detach().cpu().numpy()
        distances = args.eta * duration_matrix_np + distance_matrix_np

        # EMD
        mu_ = np.ones(t_bar_emb_rups.size(0))
        mu_ = mu_ / np.sum(mu_)
        nu_ = np.ones(t_bar_emb_rups.size(0))
        nu_ = nu_ / np.sum(nu_)
        G0 = ot.emd(mu_, nu_, distances)
        max_indices = np.argmax(G0, axis=0)
        m_shot_choose_indices = new_mu_top_indices[max_indices]
        m_shot_choose_indices_ls = m_shot_choose_indices.tolist()
        print("movie {} generated trailer index: {}".format(video_num, m_shot_choose_indices_ls))

        print(len(m_shot_choose_indices_ls))
        print(len(mu_np))
        print('generate trailers begin.')

        video_seg_concate(video_num, np.array(m_shot_choose_indices_ls), mu_np)


if __name__ == "__main__":
   movie_idx = 2  # your test movie index
   trailer_generator(movie_idx)

