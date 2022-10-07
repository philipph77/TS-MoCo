import torch
import torch.nn as nn
from .positionalEncoder import PositionalEncoder

class TransformerEncoder(nn.Module):
    """Transformer Encoder without Positonal Embeddings and withouth Tokenizer

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features: int, n_head: int, depth: int) -> None:
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.activations = [None] * len(self.transformer_encoder.layers)

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor = None) -> torch.tensor:
        # x must already be of shape (B, T, F)
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h

    def get_averaged_activations(self, K: int):
        return torch.mean(torch.stack(self.activations[max(0,len(self.activations)-K):]), dim=0)

    def hook_wrapper(self, idx):
        def hook(model, input, output):
            self.activations[idx] =  output
        return hook

class CLSTransformerEncoder(nn.Module):
    """Transformer Encoder without Postional Embeddings, Tokenizer or CLS Token

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features: int, embedding_dim: int, n_head: int, depth: int) -> None:
        super(CLSTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.activations = [None] * len(self.transformer_encoder.layers)

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor = None) -> torch.tensor:
        # x must already be of shape (B, T, F)
        cls_token = torch.randn((x.shape[0], 1, x.shape[-1]), device=x.device)
        tokens = torch.column_stack((cls_token, x))
        h = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return h, h[:,0,:]

    def get_averaged_activations(self, K: int):
        return torch.mean(torch.stack(self.activations[max(0,len(self.activations)-K):]), dim=0)

    def hook_wrapper(self, idx):
        def hook(model, input, output):
            self.activations[idx] =  output
        return hook


class CLSTokenPosTransformerEncoder(nn.Module):
    """Transformer Encoder with Postional Embeddings, Tokenizer and CLS Token

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features: int, embedding_dim: int, n_head: int, depth: int) -> None:
        super(CLSTransformerEncoder, self).__init__()
        self.tokenizer = nn.Linear(input_features, embedding_dim)
        self.positional_encoding_layer = PositionalEncoder(d_model=embedding_dim, dropout=0.1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.activations = [None] * len(self.transformer_encoder.layers)

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor = None) -> torch.tensor:
        # x: (B, C, T)
        x = x.transpose(1,2) # transpose x to shape (B, T, C)
        tokens = self.tokenizer(x) # shape (B, T, F)
        cls_token = torch.randn((tokens.shape[0], 1, tokens.shape[-1]), device=tokens.device)
        tokens = torch.column_stack((cls_token, tokens))
        tokens = self.positional_encoding_layer(tokens)
        h = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return h, h[:,0,:]

    def get_averaged_activations(self, K: int):
        return torch.mean(torch.stack(self.activations[max(0,len(self.activations)-K):]), dim=0)

    def hook_wrapper(self, idx):
        def hook(model, input, output):
            self.activations[idx] =  output
        return hook


class TokenPosTransformerEncoder(nn.Module):
    """Transformer Encoder with Postional Embeddings, Tokenizer and CLS Token

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features: int, embedding_dim: int, n_head: int, depth: int) -> None:
        super(TokenPosTransformerEncoder, self).__init__()
        self.tokenizer = nn.Linear(input_features, embedding_dim)
        self.positional_encoding_layer = PositionalEncoder(d_model=embedding_dim, dropout=0.1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.activations = [None] * len(self.transformer_encoder.layers)

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor = None) -> torch.tensor:
        # x: (B, C, T)
        x = x.transpose(1,2) # transpose x to shape (B, T, C)
        tokens = self.tokenizer(x) # shape (B, T, F)
        tokens = self.positional_encoding_layer(tokens)
        h = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return h, h[:,0,:]

    def get_averaged_activations(self, K: int):
        return torch.mean(torch.stack(self.activations[max(0,len(self.activations)-K):]), dim=0)

    def hook_wrapper(self, idx):
        def hook(model, input, output):
            self.activations[idx] =  output
        return hook