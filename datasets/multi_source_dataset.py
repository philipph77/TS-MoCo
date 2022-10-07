import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MultiSourceDataset(Dataset):
    def __init__(self, datapath, val_split_method="random"):
        super().__init__()
        assert val_split_method in ["random", "subject"]
        dataset = torch.load(datapath)
        self.X = [torch.tensor(x, dtype=torch.float) for x in dataset['X_train']]
        self.Y = [torch.tensor(y, dtype=torch.long)+1 for y in dataset['Y_train']]
        self.S = [torch.tensor(s, dtype=torch.long) for s in dataset['S_train']]
        self.train_indices, self.val_indices = self._train_val_split(val_split_method)

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [x[idx] for x in self.X], [y[idx] for y in self.Y]

    def _train_val_split(self, val_split_method):
        if val_split_method == "random":
            train_indices, val_indices = train_test_split(torch.arange(0, len(self.X[0])), test_size=0.25, random_state=42, shuffle=True, stratify=self.Y[0])
            return train_indices, val_indices
        elif val_split_method == "subject":
            train_indices, val_indices = [], []
            val_subjects = [s[0] for s in self.S]
            for s, val_subject in zip(self.S, val_subjects):
                train_indices.append(np.where(np.invert(s==val_subject))[0].tolist())
                val_indices.append(np.where(s==val_subject)[0].tolist())
            assert train_indices == [train_indices[0]]*len(train_indices)
            assert val_indices == [val_indices[0]]*len(val_indices)
            return train_indices[0], val_indices[0]


class MultiSourceDataModule(pl.LightningDataModule):
    def __init__(self, datapath, val_split_method, batch_size, num_workers):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = MultiSourceDataset(self.datapath, val_split_method)
    
    def setup(self, stage):
        self.trainset = Subset(self.dataset, self.dataset.train_indices)
        self.valset = Subset(self.dataset, self.dataset.val_indices)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    datapath = "../Datasets/multisource_train.pt"
    #dataset = MultiSourceDataset(datapath)
    datamodule = MultiSourceDataModule(datapath, 64, 1)
    datamodule.setup(0)
    train_dl = datamodule.train_dataloader()
    for x_list, y_list in train_dl:
        print([x.shape for x in x_list])
        print([y.shape for y in y_list])
        break