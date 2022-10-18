from turtle import forward
import torch
import torch.nn as nn
from architectures.transformer import TransformerEncoder
from .customLayers import TemporalSplit, OnetoManyGRU

class TSMC(nn.Module):
    def __init__(self, input_features: int, embedding_dim: int, n_head_token_enc: int, n_head_context_enc: int, depth_context_enc: int, max_predict_len: int) -> None:
        super(TSMC, self).__init__()
        self.max_predict_len = max_predict_len
        self.token_encoder = TransformerEncoder(
            use_tokenizer=True,
            use_cls_token=False,
            use_pos_embedding=True,
            input_features=input_features,
            embedding_dim=embedding_dim,
            n_head=n_head_token_enc,
            depth=1
        )
        self.context_encoder = TransformerEncoder(
            use_tokenizer=False,
            use_cls_token=True,
            use_pos_embedding=False,
            input_features=embedding_dim,
            embedding_dim=embedding_dim,
            n_head=n_head_context_enc,
            depth=depth_context_enc
        )
        self.temporal_split = TemporalSplit()
        self.prediction_head = OnetoManyGRU(embedding_dim, embedding_dim, batch_first=True)

    def forward(self, x: torch.tensor, K: int):
        tokens, _ = self.token_encoder(x)
        signal, target = self.temporal_split(tokens, K)
        _, context = self.context_encoder(signal)
        prediction = self.prediction_head(context, K)

        return context, prediction, target