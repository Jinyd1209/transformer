"""
@Author: yidong jin
@Email: yidong4242@gmail.com
"""


import torch.nn as nn

class Feed_Forward(nn.Module):
    def __init__(self, d_model, d_ff, p_prob):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(p=p_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer2(out)
        return out