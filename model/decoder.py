"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""

import torch
import torch.nn as nn
from mutil_head_attention import Mutil_Head_Attention
from layer_norm import Layer_Norm
from feed_forward import Feed_Forward

class Decoder_layer(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, p_prob, mask):
        super().__init__()
        self.mask_attention = Mutil_Head_Attention(d_model, d_k, d_v, heads, mask=mask)
        self.norm1 = Layer_Norm(d_model)
        self.drop1 = nn.Dropout(p=p_prob)
        self.attention = Mutil_Head_Attention(d_model, d_k, d_v, heads)
        self.norm2 = Layer_Norm(d_model)
        self.drop2 = nn.Dropout(p=p_prob)
        self.ffn = Feed_Forward(d_model, d_ff, p_prob)
        self.norm3 = Layer_Norm(d_model)
        self.drop3 = nn.Dropout(p=p_prob)

    def forward(self, x_encoder, x_decoder):
        identity_dec = x_decoder
        out = self.mask_attention(x_decoder, x_decoder, x_decoder)
        out = self.norm1(torch.add(identity_dec, self.drop1(out)))
        identity_1 = out
        out = self.attention(out, x_encoder, x_encoder)
        out = self.norm2(torch.add(identity_1, self.drop2(out)))
        identity_2 = out
        out = self.ffn(out)
        output = self.norm3(torch.add(identity_2, self.drop3(out)))
        return output


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, p_prob, mask, n_layers):
        super().__init__()
        self.decoder = nn.ModuleList(
            [Decoder_layer(d_model, d_ff, heads, d_k, d_v, p_prob, mask) for _ in range(n_layers)])

    def forward(self, x_enc, x_dec):
        for layer in self.decoder:
            x_dec = layer(x_enc, x_dec)

        return x_dec