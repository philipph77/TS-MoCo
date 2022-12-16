import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DREAMERDataset(Dataset):
    def __init__(self, datapath, preprocessing="None"):
        super().__init__()
        assert preprocessing in ["None", "standardize", "normalize"]
        dataset = torch.load(datapath)
        self.X_train = dataset['X_train'].float().swapaxes(1,2)
        self.X_val = dataset['X_val'].float().swapaxes(1,2)
        self.X_test = dataset['X_test'].float().swapaxes(1,2)
        self.Y_train = dataset['Y_train'].long().squeeze()#-1
        self.Y_val = dataset['Y_val'].long().squeeze()#-1
        self.Y_test = dataset['Y_test'].long().squeeze()#-1
        if preprocessing=="normalize":
            self._normalize()
        elif preprocessing=="standardize":
            self._standardize()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
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

class DREAMERDataModule(pl.LightningDataModule):
    def __init__(self, datapath, preprocessing, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = DREAMERDataset(self.datapath, preprocessing)
        self.input_features = 14
        self.n_classes = 5
    
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
    datapath = "../Datasets/DREAMER/DREAMER.pt"
    datamodule = DREAMERDataModule(datapath, "None", 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, y in trainloader:
        print(x.shape)
        print(y.shape)
        break

    dataset = DREAMERDataset(datapath, "None")
    print(torch.unique(dataset.Y_train))
    print(torch.unique(dataset.Y_val))
    print(torch.unique(dataset.Y_test))