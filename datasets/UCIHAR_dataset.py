import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class UCIHARDataModule(pl.LightningDataModule):
    def __init__(self, datapath, preprocessing, batch_size, num_workers):
        super().__init__()
        self.train_data = torch.load(os.path.join(datapath, "train_stratified.pt"))
        self.val_data = torch.load(os.path.join(datapath, "val_stratified.pt"))
        self.test_data = torch.load(os.path.join(datapath, "test_stratified.pt"))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_features = 9
        self.n_classes = 6
        self.class_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

        
        self.X_train = self.train_data['samples'].float()
        self.Y_train = self.train_data['labels'].long()
        
        self.X_val = self.val_data['samples'].float()
        self.Y_val = self.val_data['labels'].long()

        self.X_test = self.test_data['samples'].float()
        self.Y_test = self.test_data['labels'].long()

        if preprocessing=="normalize":
            self._normalize()
        elif preprocessing=="standardize":
            self._standardize()
    
    def setup(self, stage):
        self.trainset = TensorDataset(self.X_train, self.Y_train)
        self.valset = TensorDataset(self.X_val, self.Y_val)
        self.testset = TensorDataset(self.X_test, self.Y_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size, num_workers=self.num_workers, pin_memory=True)

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


if __name__ == "__main__":
    datapath = "../Datasets/UCIHAR"
    datamodule = UCIHARDataModule(datapath, 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, y in trainloader:
        print(x.shape)
        print(y.shape)
        print(torch.unique(y))
        print(len(trainloader))
        print(len(datamodule.X_train))
        break