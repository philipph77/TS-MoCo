import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class SEEDDataset(Dataset):
    def __init__(self, datapath, val_split_method="random", preprocessing="None", label="emotion"):
        super().__init__()
        assert val_split_method in ["random", "subject"]
        assert preprocessing in ["None", "standardize", "normalize"]
        assert label in ["emotion", "userID"]
        dataset = torch.load(datapath)
        testset = torch.load(datapath.replace("train", "test"))
        self.X = dataset['X_train'].float()
        self.Y = dataset['Y_train'].long()
        self.S = dataset['S_train'].long()-1
        self.X_test = testset['X_test'].float()
        self.Y_test = testset['Y_test'].long()
        self.S_test = testset['S_test'].long()-1
        self.train_indices, self.val_indices = self._train_val_split(val_split_method)
        self.label = label
        if preprocessing=="normalize":
            self._normalize()
        elif preprocessing=="standardize":
            self._standardize()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.label == "emotion":
            return self.X[idx], self.Y[idx]
        elif self.label == "userID":
            return self.X[idx], self.S[idx]
        else:
            raise ValueError(f"label must be one of ['emotion','userID'], but got {self.label}")

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
        minimum = torch.min(self.X)
        maximum = torch.max(self.X)
        self.X = (self.X - minimum) / (maximum - minimum)
        self.X_test = (self.X_test - minimum) / (maximum - minimum)

    def _standardize(self):
        mean = torch.mean(self.X)
        std = torch.std(self.X)
        self.X = (self.X-mean) / std
        self.X_test = (self.X_test-mean) / std

class SEEDDataModule(pl.LightningDataModule):
    def __init__(self, datapath, val_split_method, preprocessing, label, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SEEDDataset(self.datapath, val_split_method, preprocessing, label)
        self.input_features = 62
        if label == "emotion":
            self.n_classes = 3
            self.class_names = ["Negative", "Neutral", "Positive"]
        elif label == "userID":
            self.n_classes = 15
            self.class_name = list(range(15))
        else:
            raise ValueError(f"label must be one of ['emotion','userID'], but got {self.label}")
    
    def setup(self, stage):
        self.trainset = Subset(self.dataset, self.dataset.train_indices)
        self.valset = Subset(self.dataset, self.dataset.val_indices)
        self.testset = TensorDataset(self.dataset.X_test, self.dataset.Y_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "../Datasets/SEED_full_train.pt"
    #dataset = SEEDDataset(datapath, "random")
    datamodule = SEEDDataModule(datapath, "random", "None", "userID", 64, 1)
    datamodule.setup(None)
    trainloader = datamodule.train_dataloader()
    for x, y in trainloader:
        print(x.shape)
        print(y.shape)
        break

    dataset = SEEDDataset(datapath, "random", "None", "userID")
    print(torch.unique(dataset.S))
    print(torch.unique(dataset.S_test))
    print(torch.unique(dataset.Y))
    print(torch.unique(dataset.Y_test))