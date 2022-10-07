import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TemporalSplit(nn.Identity):
    def __init__(self):
        super(TemporalSplit, self).__init__()
        self.layer = nn.Identity()

    def forward(self, input, K):
        return self.layer(input[:,:K,:]), self.layer(input[:,K:,:])