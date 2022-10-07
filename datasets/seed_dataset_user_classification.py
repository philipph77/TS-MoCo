import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SEEDUserClassificationDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        dataset = torch.load(datapath)
        self.X = dataset['X_train'].float()
        self.S = dataset['S_train'].long() - 1
        self.session = dataset['session_train'].long()
        self.train_indices, self.val_indices = self._train_val_split()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]

    def _train_val_split(self):
        train_indices = (self.session==0).nonzero().squeeze()
        val_indices = (self.session==1).nonzero().squeeze()
        return train_indices, val_indices

class SEEDUserClassificationDataModule(pl.LightningDataModule):
    def __init__(self, datapath, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SEEDUserClassificationDataset(self.datapath)
    
    def setup(self, stage):
        self.trainset = Subset(self.dataset, self.dataset.train_indices)
        self.valset = Subset(self.dataset, self.dataset.val_indices)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "../Datasets/SEED_user_classification_train.pt"
    datamodule = SEEDUserClassificationDataModule(datapath, 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, s in trainloader:
        print(x.shape)
        print(s.shape)
        break