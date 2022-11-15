import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SEEDIJCAIDataset(Dataset):
    def __init__(self, datapath, preprocessing="None"):
        super().__init__()
        assert preprocessing in ["None", "standardize", "normalize"]
        trainset = torch.load(os.path.join(datapath, 'train.pt'))
        valset = torch.load(os.path.join(datapath, 'val.pt'))
        testset = torch.load(os.path.join(datapath, 'test.pt'))
        self.X_train = trainset['samples'].float()
        self.Y_train = trainset['labels'].long()
        self.X_val = valset['samples'].float()
        self.Y_val = valset['labels'].long()
        self.X_test = testset['samples'].float()
        self.Y_test = testset['labels'].long()
        if preprocessing=="normalize":
            self._normalize()
        elif preprocessing=="standardize":
            self._standardize()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def _normalize(self):
        minimum = torch.min(self.X)
        maximum = torch.max(self.X)
        self.X = (self.X - minimum) / (maximum - minimum)
        self.X_test = (self.X_test - minimum) / (maximum - minimum)

    def _standardize(self):
        mean = torch.mean(self.X)
        std = torch.std(self.X)
        self.X = (self.X-mean) / std
        self.X_test = (self.X_test-mean) / std

class SEEDIJCAIDataModule(pl.LightningDataModule):
    def __init__(self, datapath, preprocessing, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SEEDIJCAIDataset(self.datapath, preprocessing)
        self.input_features = 62
        self.n_classes = 3
        self.class_names = ["Negative", "Neutral", "Positive"]
    
    def setup(self, stage):
        self.trainset = TensorDataset(self.dataset.X_train, self.dataset.Y_train)
        self.valset = TensorDataset(self.dataset.X_val, self.dataset.Y_val)
        self.testset = TensorDataset(self.dataset.X_test, self.dataset.Y_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "../Datasets/SEED/"
    datamodule = SEEDIJCAIDataModule(datapath, "None", 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, y in trainloader:
        print(x.shape)
        print(y.shape)
        break

    dataset = SEEDIJCAIDataset(datapath, "None")
    print(torch.unique(dataset.Y_train))
    print(torch.unique(dataset.Y_test))
    print(torch.unique(dataset.Y_test))