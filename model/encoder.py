"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""


import torch
import torch.nn as nn
from mutil_head_attention import Mutil_Head_Attention
from layer_norm import Layer_Norm
from feed_forward import Feed_Forward

class Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, p_prob):
        super().__init__()
        self.attention = Mutil_Head_Attention(d_model, d_k, d_v, heads)
        self.norm1 = Layer_Norm(d_model)
        self.drop1 = nn.Dropout(p=p_prob)
        self.ffn = Feed_Forward(d_model, d_ff, p_prob)
        self.norm2 = Layer_Norm(d_model)
        self.drop2 = nn.Dropout(p=p_prob)

    def forward(self, x):
        identity_1 = x
        out = self.attention(x, x, x)
        out = self.norm1(torch.add(identity_1, self.drop1(out)))
        identity_2 = out
        out = self.ffn(out)
        output = self.norm2(torch.add(identity_2, self.drop2(out)))
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, p_prob, n_layers):
        super().__init__()
        self.encoder = self.make_layers(d_model, d_ff, heads, d_k, d_v, p_prob, Encoder_Layer, n_layers)

    @staticmethod
    def make_layers(d_model, d_ff, heads, d_k, d_v, p_prob, layer, n_layers):
        layers = [layer(d_model, d_ff, heads, d_k, d_v, p_prob) for _ in range(n_layers)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        return out
