from turtle import forward
import torch
import torch.nn as nn
from architectures.transformer import TokenPosTransformerEncoder, CLSTransformerEncoder, TransformerEncoder
from .customLayers import TemporalSplit, OnetoManyGRU


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
        self.prediction_head = OnetoManyGRU(embedding_dim, batch_first=True)

    def forward(self, x: torch.tensor, K: int):
        tokens, _ = self.token_encoder(x)
        signal, target = self.temporal_split(tokens, K)
        _, context = self.context_encoder(signal)
        if target.shape[1] > 0:
            prediction = self.prediction_head(context, K)
        else:
            prediction = None

        return context, prediction, target