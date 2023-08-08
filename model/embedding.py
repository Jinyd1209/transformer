"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""

import torch
import torch.nn as nn


class Token_Embedding(nn.Embedding):
    def __init__(self, voc_num, d_model):
        super().__init__(voc_num, d_model, padding_idx=0)


class Position_Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        batch, seq_len = x.size()
        # 位置编码不需要参与梯度更新
        pos_encoding = torch.zeros((seq_len, self.d_model), requires_grad=False)
        idx = torch.arange(0, seq_len).unsqueeze(dim=1)
        col_2_interval = torch.arange(0, self.d_model, 2)
        # 遍历矩阵每一个值也可以但太low，完全没有利用矩阵运算的并行特性，不考虑使用
        # TODO:是否可以统一奇偶维度？
        if self.d_model % 2 == 0:
            pos_encoding[:, 0::2] = torch.sin(idx / 10000 ** (col_2_interval / self.d_model)).float()
            pos_encoding[:, 1::2] = torch.cos(idx / 10000 ** (col_2_interval / self.d_model)).float()
        else:
            pos_encoding[:, 0::2] = torch.sin(idx / 10000 ** (col_2_interval / self.d_model)).float()
            pos_encoding[:, 1::2] = torch.cos(idx / 10000 ** (col_2_interval / self.d_model))[:, :-1].float()
        return pos_encoding


class Embedding(nn.Module):
    def __init__(self, voc_num, d_model, p_drop):
        super().__init__()
        self.token_emb = Token_Embedding(voc_num, d_model)
        self.pos_emb = Position_Encoder(d_model)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x):
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop(token_emb + pos_emb)
