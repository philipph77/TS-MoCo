import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import pytorch_lightning as pl
import torchmetrics

class plClassificationModule(pl.LightningModule):
    def __init__(self, encoder, classifier, batch_size, lr=1e-4, num_workers=0, freeze_encoder=True):
        super().__init__()
        #self.example_input_array = torch.randn(size=(batch_size, 62, 400))
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classifier = classifier
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
        if self.classifier.classifier.out_features == 1:
            self.criterion = nn.MSELoss()
            self.train_metric = torchmetrics.R2Score()
            self.val_metric = torchmetrics.R2Score()
            self.test_metric = torchmetrics.R2Score()
            self.metric_name = "R2"
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.train_metric = torchmetrics.Accuracy()
            self.val_metric = torchmetrics.Accuracy()
            self.test_metric = torchmetrics.Accuracy()
            self.metric_name = "acc"
        self.save_hyperparameters(ignore=['encoder', 'classifier'])

    def forward(self, x):
        z, _, _ = self.encoder(x, 0)
        if self.freeze_encoder:
            z = z.detach()
        return self.classifier(z)

    def configure_optimizers(self):
        return Adam(self.classifier.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        # y = y[torch.randperm(y.shape[0])] # only use for debugging purpose (shuffles the labels)
        z, _, _ = self.encoder(x, 0)
        if self.freeze_encoder:
            z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.train_metric(logits, y)
        self.log("train_loss", loss)
        self.log(f"train_{self.metric_name}", self.train_metric, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        z, _, _ = self.encoder(x, 0)
        if self.freeze_encoder:
            z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.val_metric(logits, y)

        self.log("val_loss", loss)
        self.log(f"val_{self.metric_name}", self.val_metric, prog_bar=True)
        self.log("hp_metric", self.val_metric)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        z, _, _ = self.encoder(x, 0)
        if self.freeze_encoder:
            z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.test_metric(logits, y)
        self.log(f"test_{self.metric_name}", self.test_metric, prog_bar=True)
        return {'test_loss': loss}