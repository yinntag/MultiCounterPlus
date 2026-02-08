"""PyTorch implementation of TSM with Transformer."""
import math

import numpy as np
import torch
from torch import nn
from typing import Callable
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ATSM(nn.Module):
    """ATSM model."""
    def __init__(self, clip_length: int = 64, temperature: float = 13.544):
        super(ATSM, self).__init__()
        self.clip_length = clip_length
        self.temperature = temperature
        self.temporal_context = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=(3, 1, 1), dilation=(3, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(256))
        self.tsm_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.input_projection = nn.Linear(self.clip_length * 32, 256)
        self.ln = nn.LayerNorm(256)
        self.transEncoder = TransEncoder(d_model=256, n_head=4, dropout=0.2, dim_ff=256, num_layers=1)
        # self.trans_features = self._init_transformer_head(clip_length, 2048, 4, 256)

    # @staticmethod
    # def _init_transformer_head(clip_length: int, in_features: int, n_head: int, hidden_features: int) -> nn.Module:
    #     """Initialize the fully-connected head for the final output."""
    #     return nn.Sequential(
    #         TranformerLayer(in_features, n_head, hidden_features, clip_length)
    #     )

    def forward(self, xx: torch.Tensor):
        num_proposals, _ = xx.shape  #
        x = xx.to(device)
        # x = self.tsm_label(xx)
        x = x.reshape(-1, self.clip_length, 256)
        batch_size, seq_len, _ = x.shape  #

        x = x.permute(0, 2, 1).contiguous().to(device)
        x = get_sims(x, 13.544)

        tsm_embedding = x
        # Conv layer on top of the TSM
        x = self.tsm_conv(x.permute(0, 3, 1, 2).contiguous().to(device))
        # print(x.shape)
        x = x.movedim(1, 3).reshape(batch_size, seq_len, -1)  # Flatten channels into N x D x C
        x = self.ln(F.relu(self.input_projection(x))).transpose(0, 1).contiguous()  # batch, num_frame, d_model=512
        # print(x.shape)
        period_features = self.transEncoder(x).transpose(0, 1).contiguous()

        return period_features, tsm_embedding

    def tsm_label(self, ori_label):
        multi_new_label = []
        for i in range(0, ori_label.shape[0] // 64):
            indices = ori_label[i::ori_label.shape[0] // 64]
            # print(indices.shape)
            multi_new_label.append(indices)
        new_label = torch.stack(multi_new_label, dim=0)
        return new_label


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


def pairwise_l2_distance(a: torch.Tensor, b: torch.Tensor, clip_length=64):
    """
    Computes pairwise distances between all rows of a and all rows of b.
    :param a: tensor
    :param b: tensor
    :return pairwise distance
    """
    norm_a = torch.sum(torch.square(a), dim=0)
    norm_a = torch.reshape(norm_a, [-1, 1])
    norm_b = torch.sum(torch.square(b), dim=0)
    norm_b = torch.reshape(norm_b, [1, -1])
    a = torch.transpose(a, 0, 1).contiguous()
    zero_tensor = torch.zeros(clip_length, clip_length)
    dist = torch.maximum(norm_a.cuda() - 2.0 * torch.matmul(a.cuda(), b.cuda()) + norm_b.cuda(), zero_tensor.cuda())
    # dist = cos_sim(a, b)
    return dist


def cos_sim(a, b):
    tsm = torch.matmul(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
    tsm /= math.sqrt(256)
    return F.softmax(tsm, dim=-1)


def cos_sim_m(input_tensor):
    tsm = []
    for i in range(input_tensor.size(0)):
        cosine_similarity = cos_sim(input_tensor[i], input_tensor[i].permute(1, 0))
        tsm.append(cosine_similarity)

    return torch.stack(tsm, dim=0)


def get_sims(embs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates self-similarity between batch of sequence of embeddings
    :param embs: embeddings
    :param temperature: temperature
    :return self similarity tensor
    """

    batch_size = embs.shape[0]
    seq_len = embs.shape[2]
    embs = torch.reshape(embs, [batch_size, -1, seq_len])

    def _get_sims(embs: torch.Tensor):
        """
        Calculates self-similarity between sequence of embeddings
        :param embs: embeddings
        """
        dist = pairwise_l2_distance(embs, embs)
        sims = -1.0 * dist
        return sims

    sims = map_fn(_get_sims, embs)
    # sims = torch.Size[20, 64, 64]
    sims /= math.sqrt(256)
    sims = F.softmax(sims, dim=-1)
    sims = sims.unsqueeze(dim=-1)
    return sims


def map_fn(fn: Callable, elems: torch.Tensor) -> torch.Tensor:
    """
    Transforms elems by applying fn to each element unstacked on dim 0.
    :param fn: function to apply
    :param elems: tensor to transform
    :return: transformed tensor
    """

    sims_list = []
    for i in range(elems.shape[0]):
        sims_list.append(fn(elems[i]))
    sims = torch.stack(sims_list)
    return sims


def hm_sim(embs):
    temperature = 1
    tsm = torch.cdist(embs, embs, p=0) ** 2
    tsm = -((tsm) / temperature)
    tsm = (tsm - tsm.min(dim=2)[0].unsqueeze(-1)) / (
                tsm.max(dim=2)[0].unsqueeze(-1) - tsm.min(dim=2)[0].unsqueeze(-1)).add_(1.0e-8)
    return tsm