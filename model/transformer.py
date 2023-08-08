"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""


import torch.nn as nn
from embedding import Embedding
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, voc_num, d_model, d_ff, heads, d_k, d_v, p_prob, mask, enc_layers, dec_layers):
        super().__init__()
        self.emb = Embedding(voc_num, d_model, p_prob)
        self.Encoder = Encoder(d_model, d_ff, heads, d_k, d_v, p_prob, enc_layers)
        self.Decoder = Decoder(d_model, d_ff, heads, d_k, d_v, p_prob, mask, dec_layers)
        self.linear = nn.Linear(d_model, voc_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x_enc = self.emb(x)
        x_dec = self.emb(y)
        out = self.Encoder(x_enc)
        identity_enc = out
        out = self.Decoder(identity_enc, x_dec)
        out = self.linear(out)
        output = self.softmax(out)
        return output
