"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""

import torch
import torch.nn as nn


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # 输入为一个四维的矩阵，[batch,heads,seq_len,d_k or d_v]
        # 直接使用四维矩阵，充分利用矩阵并行运算
        batch, heads, seq_len, d = v.size()
        k_t = k.transpose(2, 3)
        atten_score = (q @ k_t) / torch.sqrt(torch.tensor(d))

        if mask is not None:
            assert mask.shape == atten_score.shape, 'mask shape {} is not equal to attention score shape {}.'.format(
                mask.shape, atten_score.shape)
            atten_score = atten_score.masked_fill(mask == 0, -1e6)  # 用极小值填充使得softmax后值为0

        score = self.softmax(atten_score)
        value = score @ v
        # TODO:返回score便于后续实现可视化attention系数的heatmap
        return score, v


class Mutil_Head_Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, heads, mask=None, is_visual=False):
        super().__init__()
        self.q_w = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.mask = mask
        self.is_visual = is_visual
        self.attention = Scaled_Dot_Product_Attention()
        self.lin_concat = nn.Linear(d_model, d_model)

    def split_tensor(self, x):
        batch, seq_len, d_model = x.size()
        assert d_model // self.heads == self.d_k == self.d_v, 'd_k mutilply heads not equal to d_model.'
        mutil_head_tensor = x.view(batch, seq_len, self.heads, d_model // self.heads).transpose(1, 2)
        return mutil_head_tensor

    def forward(self, q, k, v):
        # shape[batch,seq_len,d_model]
        q, k, v = self.q_w(q), self.k_w(k), self.v_w(v)

        # 按heads数拆分成四维矩阵
        mutil_head_q, mutil_head_k, mutil_head_v = self.split_tensor(q), self.split_tensor(k), self.split_tensor(v)
        score, value = self.attention(mutil_head_q, mutil_head_k, mutil_head_v, mask=self.mask)

        # concat回原来的维度，因为是四维矩阵所以直接reshape就能搞定
        batch, heads, seq_len, d = value.size()
        out = value.transpose(1, 2).contiguous().view(batch, seq_len, heads * d)
        out = self.lin_concat(out)
        if self.is_visual:
            return score, out
        return out
