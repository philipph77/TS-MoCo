import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SEEDDataset(Dataset):
    def __init__(self, datapath, val_split_method="random", normalize=False):
        super().__init__()
        assert val_split_method in ["random", "subject"]
        dataset = torch.load(datapath)
        self.X = dataset['X_train'].float()
        self.Y = dataset['Y_train'].long()
        self.S = dataset['S_train'].long()
        self.train_indices, self.val_indices = self._train_val_split(val_split_method)
        if normalize: self._normalize()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def _train_val_split(self, val_split_method):
        if val_split_method == "random":
            train_indices, val_indices = train_test_split(torch.arange(0, len(self.X)), test_size=0.25, random_state=42, shuffle=True, stratify=self.Y)
            return train_indices, val_indices
        elif val_split_method == "subject":
            train_indices, val_indices = [], []
            val_subject = self.S[0]
            train_indices.append(np.where(np.invert(self.S==val_subject))[0].tolist())
            val_indices.append(np.where(self.S==val_subject)[0].tolist())
            return train_indices[0], val_indices[0]
    
    def _normalize(self):
        self.X = F.normalize(self.X)


class SEEDDataModule(pl.LightningDataModule):
    def __init__(self, datapath, val_split_method, normalize, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SEEDDataset(self.datapath, val_split_method, normalize)
        self.input_features = 62
        self.n_classes = 3
    
    def setup(self, stage):
        self.trainset = Subset(self.dataset, self.dataset.train_indices)
        self.valset = Subset(self.dataset, self.dataset.val_indices)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "../Datasets/SEED_full_train.pt"
    #dataset = SEEDDataset(datapath, "random")
    datamodule = SEEDDataModule(datapath, "random", True, 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, y in trainloader:
        print(x.shape)
        print(y.shape)
        break