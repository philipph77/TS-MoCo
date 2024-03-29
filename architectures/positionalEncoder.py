import math
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, pos_embeddings_alpha: int=1, dropout: float=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=False):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.pos_embeddings_alpha = pos_embeddings_alpha
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        if self.batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pos_embeddings_alpha*self.pe[:,:x.size(1)]
        else:
            x = x + self.pos_embeddings_alpha*self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':
    pe = PositionalEncoder(d_model=50, batch_first=True)
    x = torch.randn(size=(64,100,50))
    x + pe(x)
    print(pe.pe[0,:,0])

    pe = PositionalEncoder(d_model=50, batch_first=False)
    x = torch.randn(size=(100,64,50))
    x + pe(x)
    print(pe.pe[:,0,0])
