from turtle import forward
import torch
import torch.nn as nn
from architectures.transformer import TokenPosTransformerEncoder, CLSTransformerEncoder, TransformerEncoder
from .customLayers import TemporalSplit


class TSMC(nn.Module):
    def __init__(self, input_features: int, embedding_dim: int, n_head_token_enc: int, n_head_context_enc: int, depth_context_enc: int, max_predict_len: int) -> None:
        super(TSMC, self).__init__()
        self.max_predict_len = max_predict_len
        self.token_encoder = TokenPosTransformerEncoder(
            input_features,
            embedding_dim,
            n_head_token_enc,
            depth=1
        )
        self.context_encoder = CLSTransformerEncoder(
            embedding_dim,
            embedding_dim,
            n_head_context_enc,
            depth_context_enc
        )
        self.temporal_split = TemporalSplit()
        self.prediction_heads = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(self.max_predict_len)])

    def forward(self, x: torch.tensor, K: int):
        tokens, _ = self.token_encoder(x)
        signal, target = self.temporal_split(tokens, K)
        _, context = self.context_encoder(signal)
        if target.shape[1] > 0:
            prediction = torch.stack([prediction_head(context) for prediction_head in self.prediction_heads][:target.shape[1]], dim=1)
        else:
            prediction = None

        return context, prediction, target