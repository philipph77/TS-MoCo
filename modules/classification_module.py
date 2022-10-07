import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import pytorch_lightning as pl
import torchmetrics

class plClassificationModule(pl.LightningModule):
    def __init__(self, encoder, classifier, batch_size, lr=1e-4, num_workers=0):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classifier = classifier
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters(ignore=['encoder', 'classifier'])

    def forward(self, x):
        z, _, _ = self.encoder(x, x.shape[2])
        return self.classifier(z)

    def configure_optimizers(self):
        return Adam(self.classifier.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        # y = y[torch.randperm(y.shape[0])] # only use for debugging purpose (shuffles the labels)
        z, _, _ = self.encoder(x, x.shape[2])
        z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.train_accuracy(logits, y)
        pbar = {'train_acc': self.train_accuracy.compute()}
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy)
        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        z, _, _ = self.encoder(x, x.shape[2])
        z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.val_accuracy(logits, y)
        pbar = {'val_acc': self.val_accuracy.compute()}
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy)
        self.log("hp_metric", self.val_accuracy)
        return {'val_loss': loss, 'progress_bar': pbar}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
        avg_val_acc = self.val_accuracy.compute()
        pbar = {'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_val_loss, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        z, _, _ = self.encoder(x, x.shape[2])
        z = z.detach()
        logits = self.classifier(z)
        loss = self.criterion(logits, y)
        self.test_accuracy(logits, y)
        pbar = {'test_acc': self.test_accuracy.compute()}
        self.log("test_acc", self.test_accuracy)
        return {'test_loss': loss, 'progress_bar': pbar}