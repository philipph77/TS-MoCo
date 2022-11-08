from os import path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class Cho2017DataModule(pl.LightningDataModule):
    def __init__(self, datapath, preprocessing, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        traindata = torch.load(path.join(datapath, 'train.pt'))
        valdata = torch.load(path.join(datapath, 'val.pt'))
        testdata = torch.load(path.join(datapath, 'test.pt'))

        self.X_train, self.Y_train = traindata['X_train'], traindata['Y_train'].long()
        self.X_val, self.Y_val = valdata['X_val'], valdata['Y_val'].long()
        self.X_test, self.Y_test = testdata['X_test'], testdata['Y_test'].long()
        self.trainset = TensorDataset(self.X_train, self.Y_train)
        self.valset = TensorDataset(self.X_val, self.Y_val)
        self.testset = TensorDataset(self.X_test, self.Y_test)
        if preprocessing=="normalize":
            self._normalize()
        elif preprocessing=="standardize":
            self._standardize()
        self.input_features = 64
        self.n_classes = 2
    
    def setup(self, stage):
        pass

    def _normalize(self):
        minimum = torch.min(self.X_train)
        maximum = torch.max(self.X_train)
        self.X_train = (self.X_train - minimum) / (maximum - minimum)
        self.X_val = (self.X_val - minimum) / (maximum - minimum)
        self.X_test = (self.X_test - minimum) / (maximum - minimum)

    def _standardize(self):
        mean = torch.mean(self.X_train)
        std = torch.std(self.X_train)
        self.X_train = (self.X_train-mean) / std
        self.X_val = (self.X_val-mean) / std
        self.X_test = (self.X_test-mean) / std

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "datasets/Cho2017/"
    #dataset = SEEDDataset(datapath, "random")
    datamodule = Cho2017DataModule(datapath, 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x,y in trainloader:
        print(x.shape)
        print(y)
        break