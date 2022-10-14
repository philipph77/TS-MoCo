import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import pytorch_lightning as pl
from functions.masking import random_masking, channel_wise_masking, temporal_masking

class plEncodingModule(pl.LightningModule):
    def __init__(self, encoder, batch_size, lr=1e-4, tau=0.9, lam=1.0, masking_percentage=0.5, masking_method="random", num_workers=0):
        assert masking_method in ['random', 'channel_wise', 'temporal']
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.lam = lam
        self.masking_percentage = masking_percentage
        self.num_workers = num_workers

        for name, weight in self.teacher.named_parameters():
                weight.requires_grad = False

        self.tc_criterion = nn.CosineEmbeddingLoss(reduction="mean")
        self.cc_criterion = nn.CosineEmbeddingLoss(reduction="mean")
        
        if masking_method == 'random': self.masking_func = random_masking
        elif masking_method == 'channel_wise': self.masking_func = channel_wise_masking
        elif masking_method == 'temporal': self.masking_func = temporal_masking
        else: raise ValueError(f'masking_method must be one of random, channel_wise, temporal but got {masking_method}')
        
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x):
        return self.student(x)[0]
    
    def configure_optimizers(self):
        return Adam(self.student.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss = self.forward_iteration(batch, batch_idx, "train")
        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):            
            # Update Teacher Models Parameters
            for name, weight in self.teacher.named_parameters():
                self.teacher.state_dict()[name].copy_(
                    weight.mul_(self.tau).add_(self.student.state_dict()[name], alpha=1-self.tau)
                )

    def validation_step(self, batch, batch_idx):
        return self.forward_iteration(batch, batch_idx, "val")

    def forward_iteration(self, batch, batch_idx, mode):
        x = batch[0]
        x_masked = self.masking_func(x, self.masking_percentage)
        K = random.randint(1, self.student.max_predict_len)
        c_T, x_pred_T, x_T = self.teacher(x, K)
        c_S, x_pred_S, x_S = self.student(x_masked, K)

        loss_tc_S = self.tc_criterion(torch.flatten(x_pred_S, start_dim=1), torch.flatten(x_T, start_dim=1), torch.ones(size=(x.shape[0],), device=x.device))
        loss_tc_T = self.tc_criterion(torch.flatten(x_pred_T, start_dim=1), torch.flatten(x_S, start_dim=1), torch.ones(size=(x.shape[0],), device=x.device))
        loss_cc = self.cc_criterion(torch.flatten(c_S, start_dim=1), torch.flatten(c_T, start_dim=1), torch.ones(size=(x.shape[0],), device=x.device))

        loss = loss_tc_S + loss_tc_T + self.lam*loss_cc
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_loss_tc_S", loss_tc_S)
        self.log(f"{mode}_loss_tc_T", loss_tc_T)
        self.log(f"{mode}_loss_cc", loss_cc)
        return loss