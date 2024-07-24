import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import skvideo.io
import torch.nn.functional as F
from model import VA_encoder_self_cross_sigmoid
import os.path as osp
import argparse
import torch.optim as optim
import random
from torch.nn.functional import normalize
from sinkhorn import SinkhornDistance
from utils import fix_seeds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='5')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--delta', default=1.0, type=float, help='relaxtion parameter')
    parser.add_argument('--lamda', default=1.0, type=float,
                        help='parameter controlling the regularity of distance for pi')
    parser.add_argument('--n_epochs', type=int, default=4000)
    parser.add_argument("--exp", type=str, default='', help="save dir name")
    parser.add_argument("--trailer_audio_base", type=str, default='path of trailer audio base')
    parser.add_argument("--trailer_shot_base", type=str, default='path of trailer shot base')
    parser.add_argument("--movie_shot_base", type=str, default='path of movie shot base')
    parser.add_argument("--mv_pretrained_trailer_audio_base", type=str, default='path of mv dataset trailer audio base')
    parser.add_argument("--mv_pretrained_movie_shot_base", type=str, default='path of mv dataset movie shot base')
    args = parser.parse_args()

    return args


def train():
    fix_seeds(42)
    args = get_args()

    # GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = args.device

    model = VA_encoder_self_cross_sigmoid(input_dim=args.input_size, model_dim=args.hidden_size).to(device)
    model.train()
    exp = "VA_encoder_self_cross_sigmoid_epochs={}_lr={}_hidden={}_delta={}_lamda={}_pretrained_lr={}_partial".format(
        args.n_epochs, args.lr, args.hidden_size, args.delta, args.lamda, 0.0001)

    save_model_base = 'path to save model'
    save_model_base = osp.join(save_model_base, exp)
    os.makedirs(save_model_base, exist_ok=True)

    videos = os.listdir(args.trailer_shot_base)

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs * len(videos))

    loss_set = []

    for epoch in range(1, args.n_epochs + 1):
        loss_ = []

        for video in videos:
            # e.g. video = 1-0.npy
            trailer_idx = video[:-4]  # trailer_idx = 1-0
            movidx = trailer_idx.split('-')[0]
            print('trailer_idx: {}'.format(trailer_idx))

            m_shot = torch.Tensor(np.load(osp.join(args.movie_shot_base, movidx + '.npy'))).to(device)
            t_bar = torch.Tensor(np.load(osp.join(args.trailer_audio_base, video))).to(device)

            # normalize embeddings
            m_shot = normalize(m_shot, p=2.0, dim=1)
            t_bar = normalize(t_bar, p=2.0, dim=1)

            m_shot_emb, mu_hat, t_bar_emb = model(m_shot, t_bar)

            # normalize embeddings
            m_shot_emb = normalize(m_shot_emb, p=2.0, dim=1)
            t_bar_emb = normalize(t_bar_emb, p=2.0, dim=1)

            # obtain corresponding relation between m_shot and a_bar: pi_hat
            b_mt_relation_path = "path of movie-trailer shot alignment file"
            with open(os.path.join(b_mt_relation_path, "{}.json".format(trailer_idx)), 'r') as f:
                b_mt_relation = json.load(f)
            coord = [(value[0][1], int(key)) for key, value in
                     b_mt_relation.items()]  # selected shot index and its corresponding audio shot index, (selected shot index, audio shot index)
            coord_movie_index = [value[0][1] for key, value in b_mt_relation.items()]  # selected shot index

            # emperical matrix: pi and mu
            pi = torch.zeros((m_shot_emb.size(0), t_bar_emb.size(0))).to(device)
            for coo in coord:
                x, y = coo
                pi[x, y] = 1
            mu = torch.sum(pi, dim=1).unsqueeze(1)  # selected shot => 1, unselected shot => 0
            nu = torch.sum(pi, dim=0).unsqueeze(1)
            nu = nu / torch.sum(nu)  # nu is uniform distribution

            # choose partial movie shot
            mu_hat_top_values, mu_hat_top_indices = torch.topk(mu_hat.view(-1), k=t_bar_emb.size(0), largest=True)
            choose_movie_shot = m_shot_emb[mu_hat_top_indices]

            # emperical matrix: partial_pi
            partial_pi = torch.zeros((choose_movie_shot.size(0), t_bar_emb.size(0))).to(device)
            for value in mu_hat_top_indices:
                if value in coord_movie_index:
                    mu_hat_index = torch.where(mu_hat_top_indices == value)
                    coord_index = coord_movie_index.index(value)
                    x, y = coord[coord_index]
                    partial_pi[mu_hat_index, y] = 1

            partial_pi += 1e-7
            partial_pi /= partial_pi.sum()
            partial_mu_hat = torch.ones(t_bar_emb.size(0)).to(device)
            partial_mu_hat = partial_mu_hat / torch.sum(partial_mu_hat)  # partial_mu_hat is uniform distribution

            # compute transport plan: partial_pi_hat (partial_mu_hat and nu are uniform distributions)
            sinkhorn = SinkhornDistance(lamda=args.lamda)
            d_pi, partial_pi_hat, C = sinkhorn(x=choose_movie_shot, y=t_bar_emb, margin_mu=partial_mu_hat, margin_nu=nu,
                                               device=device)

            # KL loss
            loss_pi = torch.sum(partial_pi_hat * torch.log(partial_pi_hat / partial_pi))
            # BCE loss
            loss_mu = F.binary_cross_entropy(mu_hat, mu)
            loss = loss_pi + args.delta * loss_mu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_.append(loss.item())

        loss_set.append(np.mean(loss_))

        print('-' * 100)
        print('Epoch {}:'.format(epoch))
        print('loss: {}'.format(np.mean(loss_)))
        print('loss_set: {}'.format(loss_set))
        print('-' * 100)

        if epoch % 100 == 0:
            # save model parameters
            save_path = osp.join(save_model_base, 'network_{}.net'.format(epoch))
            torch.save(model.state_dict(), save_path)

    # plot loss figures
    x = np.arange(1, args.n_epochs + 1)
    y1 = np.array(loss_set)
    plt.plot(x, y1, 'green', label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    fig_dir = os.path.join(save_model_base, 'loss.png')
    plt.savefig(fig_dir)
    plt.close('all')


if __name__ == "__main__":
    print("Training start!")
    train()
