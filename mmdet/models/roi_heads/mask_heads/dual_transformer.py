import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]

        I = torch.ones(f).to(self.device)
        # I = torch.ones(f)
        xr = torch.einsum('bfe,h->bhfe', (x, I))  # [x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))  # [x x x x ....]     =>  xc[:,:,0,:] == x

        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        # print(out.shape)
        # out = self.bn(out)
        out = F.softmax(-out / math.sqrt(256), dim=-1)
        return out


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


class DUAL(nn.Module):
    def __init__(self, clip_length=64):
        super(DUAL, self).__init__()
        self.clip_length = clip_length
        self.sims = Sims()
        self.mha_sim = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.conv3x3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.input_projection = nn.Linear(self.clip_length * 32, 256)
        self.ln = nn.LayerNorm(256)
        self.transEncoder = TransEncoder(d_model=256, n_head=4, dropout=0.2, dim_ff=256, num_layers=3)

    def forward(self, xx):
        num_proposals, _ = xx.shape  #
        # x = self.tsm_label(xx)
        x = xx.reshape(-1, self.clip_length, 256)
        batch_size, seq_len, _ = x.shape
        att1 = F.relu(self.sims(x))
        # print(att1.shape)
        x = x.transpose(0, 1).contiguous()
        _, att2 = self.mha_sim(x, x, x)
        att2 = F.relu(att2.unsqueeze(1))
        # print(att2.shape)
        # att = torch.cat([att1, att2], dim=1)
        att = att1 + att2
        # print(att.shape)
        # np.save("/mnt/tbdisk/tangyin/DeepCount/demo_video/intermediate_results/msa.npy", att.cpu().detach().numpy())
        y = self.dropout(F.relu(self.bn(self.conv3x3(att)))).permute(0, 2, 3, 1).contiguous()
        # print(y.shape)
        y = y.reshape(batch_size, self.clip_length, -1)  # batch, num_frame, 32*num_frame
        y = self.ln(F.relu(self.input_projection(y))).transpose(0, 1).contiguous()  # [64, bs, 256]
        # y = self.ln(F.relu(self.input_projection(y)))
        y = self.transEncoder(y).transpose(0, 1).contiguous()
        return y, att


# if __name__ == '__main__':
#     model = DUAL().cuda()
#     x = torch.randn(2, 64, 256).cuda()
#     out = model(x)
#     print(out[0].shape)