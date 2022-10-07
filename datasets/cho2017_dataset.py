from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from braindecode.datautil import load_concat_dataset

class Cho2017DataModule(pl.LightningDataModule):
    def __init__(self, datapath, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset = load_concat_dataset(path.join(datapath, 'train'), preload=True)
        self.valset = load_concat_dataset(path.join(datapath, 'val'), preload=True)
        #self.test = load_concat_dataset(path.join(datapath, 'test'), preload=True)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "datasets/Cho2017/"
    #dataset = SEEDDataset(datapath, "random")
    datamodule = Cho2017DataModule(datapath, 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    #valloader = datamodule.val_dataloader()
    for x,y,_ in trainloader:
        print(x.shape)
        print(y)
        break