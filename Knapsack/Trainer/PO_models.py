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
from Trainer.diff_layer import SPOlayer, DBBlayer

from DPO import perturbations
from DPO import fenchel_young as fy
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

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

class DBB(twostage_mse):
    def __init__(self,weights,capacity,n_items,lambda_val=1., lr=1e-1,seed=1):
        super().__init__(weights,capacity,n_items,lr,seed)
        self.layer = DBBlayer(self.solver, lambda_val=lambda_val)
    

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        solver = self.solver
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(y_hat, y,sol ) 
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class FenchelYoung(twostage_mse):
    def __init__(self,weights,capacity,n_items,sigma=0.1,num_samples=10, lr=1e-1,seed=1):
        super().__init__(weights,capacity,n_items,lr,seed)  

        fy_solver =  lambda y_: batch_solve(self.solver,y_) 
        self.criterion = fy.FenchelYoungLoss(fy_solver, num_samples= num_samples, 
        sigma= sigma,maximize = True, batched= True)
    def training_step(self, batch, batch_idx):
        criterion = self.criterion 
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = criterion(y_hat,sol).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class IMLE(twostage_mse):
    def __init__(self,weights,capacity,n_items, k=5, nb_iterations=100,nb_samples=1, 
            input_noise_temperature=1.0, target_noise_temperature=1.0,  lr=1e-1,seed=1):
        super().__init__(weights,capacity,n_items,lr,seed)
        imle_solver = lambda y_: batch_solve(self.solver,y_)

        target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
        noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations= nb_iterations)

        self.layer = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= input_noise_temperature, target_noise_temperature= target_noise_temperature,
                    nb_samples= nb_samples)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        solver = self.solver
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(y_hat ) 
        # print("shape of sol")
        # print(sol_hat.shape, sol.shape)
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss