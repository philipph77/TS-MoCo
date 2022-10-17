import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TemporalSplit(nn.Identity):
    def __init__(self):
        super(TemporalSplit, self).__init__()
        self.layer = nn.Identity()

    def forward(self, input, K):
        return self.layer(input[:,:-K,:]), self.layer(input[:,-K:,:])

class OnetoManyGRU(nn.Module):
    def __init__(self, embedding_dim: int, batch_first: bool = True):
        super(OnetoManyGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_first = batch_first
        self.prediction_head = nn.GRU(embedding_dim, embedding_dim, batch_first=batch_first)

    def forward(self, c: torch.Tensor, K: int) -> torch.Tensor:
        if self.batch_first:
            batch_size = c.size(0)    
            x_k = torch.zeros(batch_size, 1, self.embedding_dim)
        else:
            batch_size = c.size(1)
            x_k = torch.zeros(1, batch_size, self.embedding_dim)
        h_k = c.unsqueeze(0)

        y_out = []
        for k in range(K):
            y_k, h_k = self.prediction_head(x_k, h_k)
            y_out.append(y_k)
            x_k = y_k
        
        return torch.cat(y_out, dim=1)