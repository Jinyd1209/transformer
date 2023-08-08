"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""

import torch
import torch.nn as nn

class Layer_Norm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # 为了和pytorch的官方实现保持一致，采用无偏估计计算方差
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.alpha * out + self.beta
        return out
