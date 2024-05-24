import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.Encoder import Encoder
from models.ETC import ETC
from models.UWE import UWE


class CFDTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(args)
        self.train_time_wordcount = args.train_time_wordcount

        self.a = 1 * np.ones((1, args.num_topic)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T + (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(args.vocab_size, affine=False)

        self.word_embeddings = torch.from_numpy(args.word_embeddings).float()
        self.word_embeddings = nn.Parameter(self.word_embeddings)

        # topic_embeddings: TxKxD
        self.topic_embeddings = nn.init.xavier_normal_(torch.zeros(args.num_topic, self.word_embeddings.shape[1])).repeat(args.num_times, 1, 1)
        self.topic_embeddings = nn.Parameter(self.topic_embeddings)

        self.ETC = ETC(self.args.num_times, self.args.model.temperature, self.args.model.weight_neg, self.args.model.weight_pos)
        self.UWE = UWE(self.ETC, self.args.num_times, self.args.model.temperature, self.args.model.weight_UWE, self.args.model.neg_topk)

    def get_beta(self):
        dist = self.pairwise_euclidean_dist(F.normalize(self.topic_embeddings, dim=-1), F.normalize(self.word_embeddings, dim=-1))
        beta = F.softmax(-dist / self.args.model.beta_temp, dim=1)

        return beta

    def pairwise_euclidean_dist(self, x, y):
        cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, axis=-1) - 2 * torch.matmul(x, y.t())
        return cost

    def get_theta(self, x, times=None):
        theta, mu, logvar = self.encoder(x)
        if self.training:
            return theta, mu, logvar

        return theta

    def get_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.args.num_topic)

        return KLD.mean()

    def get_NLL(self, theta, beta, x, recon_x=None):
        if recon_x is None:
            recon_x = self.decode(theta, beta)
        recon_loss = -(x * recon_x.log()).sum(axis=1)

        return recon_loss

    def decode(self, theta, beta):
        d1 = F.softmax(self.decoder_bn(torch.bmm(theta.unsqueeze(1), beta).squeeze(1)), dim=-1)
        return d1

    def forward(self, x, times):
        loss = 0.

        theta, mu, logvar = self.get_theta(x)
        kl_theta = self.get_KL(mu, logvar)

        loss += kl_theta

        beta = self.get_beta()
        time_index_beta = beta[times]
        recon_x = self.decode(theta, time_index_beta)
        NLL = self.get_NLL(theta, time_index_beta, x, recon_x)
        NLL = NLL.mean()
        loss += NLL

        loss_ETC = self.ETC(self.topic_embeddings)
        loss += loss_ETC

        loss_UWE = self.UWE(self.train_time_wordcount, beta, self.topic_embeddings, self.word_embeddings)
        loss += loss_UWE

        rst_dict = {
            'loss': loss,
        }

        return rst_dict
