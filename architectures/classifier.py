import torch
import torch.nn as nn

class DenseClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseClassifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.classifier(x)

class MultiLayerDenseClassifier(nn.Module):
    def __init__(self, in_features: int, n_layers: int, out_features: int = 3) -> None:
        super(MultiLayerDenseClassifier, self).__init__()
        assert n_layers > 0
        if n_layers == 1: self.classifier = nn.Linear(in_features, out_features)
        latent_dims = torch.linspace(in_features, out_features, n_layers+1, dtype=int).tolist()
        layers = []
        for in_features, out_features in zip(latent_dims[:-1], latent_dims[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        layers = layers[:-1] #remove last ReLU layer
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.classifier(x)