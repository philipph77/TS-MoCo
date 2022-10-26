import torch
import torch.nn as nn
from .positionalEncoder import PositionalEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, use_tokenizer: bool, use_cls_token: bool, use_pos_embedding: bool, input_features: int, embedding_dim: int, n_head: int, depth: int) -> None:
        super(TransformerEncoder, self).__init__()
        if not(input_features==embedding_dim) and not use_tokenizer: raise ValueError("Tokenizer must be used if input_features does not match embedding_dim")
        self.use_tokenizer = use_tokenizer
        self.use_cls_token = use_cls_token
        self.use_pos_embedding = use_pos_embedding
        self.tokenizer = nn.Linear(input_features, embedding_dim) if self.use_tokenizer else nn.Identity()
        self.positional_encoding_layer = PositionalEncoder(d_model=embedding_dim, dropout=0.1, batch_first=True) if self.use_pos_embedding else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.activations = [None] * len(self.transformer_encoder.layers)
        if use_cls_token: self.cls_token = nn.Parameter( torch.randn(1, 1, embedding_dim) )

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor = None) -> torch.tensor:
        if self.use_tokenizer:
            # x: (B, C, T)
            x = x.transpose(1,2) # transpose x to shape (B, T, C)
            tokens = self.tokenizer(x) # shape (B, T, F)
        else:
            # x must already be of shape (B, T, F)
            tokens = self.tokenizer(x)
        if self.use_cls_token:
            tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) #tokens is of shape [B, 1+T, F]
            
        tokens = self.positional_encoding_layer(tokens)
        h = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return h, h[:,0,:]

    def get_averaged_activations(self, K: int):
        return torch.mean(torch.stack(self.activations[max(0,len(self.activations)-K):]), dim=0)

    def hook_wrapper(self, idx):
        def hook(model, input, output):
            self.activations[idx] =  output
        return hook