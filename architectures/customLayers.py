import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TemporalSplit(nn.Identity):
    def __init__(self, split_dim=1):
        super(TemporalSplit, self).__init__()
        self.layer = nn.Identity()
        self.split_dim = split_dim

    def forward(self, input, K):
        if self.split_dim == 0: return self.layer(input[:-K,:,:]), self.layer(input[-K:,:,:])
        elif self.split_dim == 1: return self.layer(input[:,:-K,:]), self.layer(input[:,-K:,:])
        elif self.split_dim == 2: return self.layer(input[:,:,:-K]), self.layer(input[:,:,-K:])
        else: raise ValueError(f"split_dim must be one of [0,1,2], but got {self.split_dim}")
        

class OnetoManyGRU(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, batch_first: bool = True):
        super(OnetoManyGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_first = batch_first
        self.prediction_head = nn.GRU(embedding_dim, embedding_dim, batch_first=batch_first)
        if embedding_dim == output_dim:
            self.transpose = False
            self.untokenizer = nn.Identity()
        else:
            self.transpose = True
            self.untokenizer = nn.Linear(embedding_dim, output_dim)

    def forward(self, c: torch.Tensor, K: int) -> torch.Tensor:
        if self.batch_first:
            batch_size = c.size(0)    
            x_k = torch.zeros(batch_size, 1, self.embedding_dim, device=c.device)
        else:
            batch_size = c.size(1)
            x_k = torch.zeros(1, batch_size, self.embedding_dim, device=c.device)
        h_k = c.unsqueeze(0).contiguous()

        y_out = []
        for k in range(K):
            y_k, h_k = self.prediction_head(x_k, h_k)
            y_out.append(y_k)
            x_k = y_k

        y = torch.cat(y_out, dim=1)
        y = self.untokenizer(y)
        if self.transpose: y = y.transpose(1,2)
        
        return y