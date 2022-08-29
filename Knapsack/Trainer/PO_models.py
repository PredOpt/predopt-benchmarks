import pandas as pd
import numpy as np 
from torch import nn, optim
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
from Trainer.comb_solver import knapsack_solver
from Trainer.utils import batch_solve, regret_fn,regret_list
from Trainer.diff_layer import SPOlayer
class twostage_mse(pl.LightningModule):
    def __init__(self,weights,capacity,n_items,lr=1e-1,seed=1):
        super().__init__()
        pl.seed_everything(seed)
        self.model = nn.Linear(8,1)
        self.lr = lr
        self.solver = knapsack_solver(weights,capacity, n_items)
        self.save_hyperparameters("lr")

    def forward(self,x):
        return self.model(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y,sol)
        mseloss = criterion(y_hat, y)

        self.log("val_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss, }
    def predict_step(self, batch, batch_idx):
        '''
        I am using the the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        regret_tensor = regret_list(solver,y_hat,y,sol)
        return regret_tensor
    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y,sol)
        mseloss = criterion(y_hat, y)

        self.log("test_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"test_regret": val_loss, "test_mse": mseloss, }
    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6),
                    "monitor": "val_regret",
                },
            }


class SPO(twostage_mse):
    def __init__(self,weights,capacity,n_items,lr=1e-1,seed=1):
        super().__init__(weights,capacity,n_items,lr,seed)
        self.layer = SPOlayer(self.solver)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        solver = self.solver
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol ) 

    
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    