
from Trainer.NNModels import cora_net, cora_normednet, cora_nosigmoidnet
from Trainer.utils import regret_fn, regret_list
from Trainer.diff_layer import *
from DPO import perturbations
from DPO import fenchel_young as fy
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
import pandas as pd

import numpy as np 
from torch import nn, optim
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl


class baseline_mse(pl.LightningModule):
    def __init__(self,solver,lr=1e-1,mode='default',seed=0):
        super().__init__()
        pl.seed_everything(seed)
        if mode=='default':
            self.model = cora_net(n_layers=2)
            
        elif mode=="batchnorm" :
            self.model = cora_normednet(n_layers=2)
        elif mode=="linear":
            self.model = cora_nosigmoidnet(n_layers=2)
        self.mode= mode

        self.lr = lr
        self.solver = solver
        self.save_hyperparameters("lr")

    def forward(self,x):
        return self.model(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        solver = self.solver
        
        x,y,sol,m = batch

        y_hat =  self(x).squeeze()
        val_loss= regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        if self.mode== "linear":
           y_hat = torch.sigmoid(y_hat)
        criterion2 = nn.BCELoss(reduction='mean')
        bceloss = criterion2(y_hat, sol)


        self.log("val_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_bce", bceloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss}

    def test_step(self, batch, batch_idx):
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        criterion2 = nn.BCELoss(reduction='mean')
        if self.mode== "linear":
           y_hat = torch.sigmoid(y_hat)

        bceloss = criterion2(y_hat, sol)
        self.log("test_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )    
        self.log("test_bce", bceloss, prog_bar=True, on_step=True, on_epoch=True, )        
        return  {"test_regret": val_loss, "test_mse": mseloss}
    def predict_step(self, batch, batch_idx):
        '''
        I am using the the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        regret_tensor = regret_list(solver,y_hat,y,sol,m)
        return regret_tensor
    def configure_optimizers(self):
        
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6),
                    "monitor": "val_regret"
            }}

class baseline_bce(baseline_mse):
    def __init__(self,solver,lr=1e-1,mode='default',seed=0):
            super().__init__(solver,lr,mode,seed) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(y_hat,sol)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss   


class SPO(baseline_mse):
    def __init__(self,solver, lr=1e-1,mode='default',seed=0):
        super().__init__(solver,lr,mode, seed)
        self.layer = SPOlayer(solver)
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB(baseline_mse):
    def __init__(self, solver,lr=1e-1,lambda_val=0.1,mode='default', seed=0):
        super().__init__(solver,lr,mode, seed)
        self.layer = DBBlayer(solver,lambda_val=lambda_val)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class FenchelYoung(baseline_mse):
    def __init__(self,solver,sigma=0.1,num_samples=10, 
        lr=1e-1,mode='default', seed=0):
        self.sigma = sigma
        self.num_samples = num_samples
        super().__init__(solver,lr,mode, seed)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()

        criterion = fy.FenchelYoungLoss(self.solver, num_samples= self.num_samples, sigma= self.sigma,maximize = True, batched= True)
        loss = criterion(y_hat, sol, m)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss



# def MAP(sol,y,solpool,minimize=False):
#     '''
#     sol, y and y_hat are torch array [batch_size,48]
#     solpool is torch array [currentpoolsize,48]
#     '''
#     mm = 1 if minimize else -1 
#     loss = 0
#     # print("shape check", sol.shape, y.shape,y_hat.shape, solpool.shape)
#     for ii in range(len(y)):
#         loss += torch.max(((sol[ii] - solpool )*(mm*y[ii]  )).sum(dim=1))
#     return loss
# def MAP_c(y_hat,y_true,sol,solpool,*wd,**kwd):
#     y = y_hat 
#     return MAP(sol,y,solpool)
# def MAP_hatc_c(y_hat,y_true,sol,solpool,*wd,**kwd):
#     y = y_hat - y_true
#     return MAP(sol,y,solpool)

# def NCE(sol,y,solpool,minimize=False):
#     '''
#     sol, y and y_hat are torch array [batch_size,48]
#     solpool is torch array [currentpoolsize,48]
#     '''
#     mm = 1 if minimize else -1 
#     loss = 0
#     # print("shape check", sol.shape, y.shape,y_hat.shape, solpool.shape)
#     for ii in range(len(y)):
#         loss += torch.mean(((sol[ii] - solpool )*(mm*y[ii]  )).sum(dim=1))
#     return loss
# def NCE_c(y_hat,y_true,sol,solpool,*wd,**kwd):
#     y = y_hat 
#     return NCE(sol,y,solpool)
# def NCE_hatc_c(y_hat,y_true,sol,solpool,*wd,**kwd):
#     y = y_hat - y_true
#     return NCE(sol,y,solpool)

# # def pointwise_loss(y_hat,y_true,sol,solpool,*wd,**kwd):
# #     '''
# #     sol, y and y_hat are torch array [batch_size,48]
# #     solpool is torch array [currentpoolsize,48]
# #     '''
# #     criterion = nn.MSELoss(reduction='mean')
# #     loss = 0
# #     for ii in range(len(y_true)):
# #         _,indices = torch.unique((y_true[ii]*solpool).sum(dim=1), sorted=True, return_inverse=True) 
# #         loss  += criterion( (y_hat[ii]*solpool).sum(dim=1) ,indices.float() )
# #     return loss


# def pointwise_loss(y_hat,y_true,sol,solpool,*wd,**kwd):
#     '''
#     sol, y and y_hat are torch array [batch_size,48]
#     solpool is torch array [currentpoolsize,48]
#     f(y_hat,s) is regresson on f(y,s)
#     '''
#     loss = (torch.matmul(y_hat,solpool.transpose(0,1))- torch.matmul(y_true,solpool.transpose(0,1))).square().mean()

#     return loss

# def pairwise_loss(y_hat,y_true,sol,solpool,margin=0, minimize=False,mode= 'B'):
#     '''
#     sol, y and y_hat are torch array [batch_size,48]
#     solpool is torch array [currentpoolsize,48]
#     '''
#     mm = 1 if minimize else -1 
#     loss = 0
#     relu = nn.ReLU()
#     for ii in range(len(y_true)):
#         _,indices= np.unique((mm*y_true[ii]*solpool).sum(dim=1).detach().numpy(),return_index=True)
#         ## return indices after sorting the array in ascending order
#         if mode == 'B':
#             big_ind = [indices[0] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
#         if mode == 'W':
#             big_ind = [indices[p] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
#         if mode == 'S':
#             big_ind = [indices[p] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one


#         loss += relu(margin+ mm*( torch.matmul(solpool[big_ind], y_hat[ii]) - torch.matmul(solpool[small_ind], y_hat[ii])) ).mean()
        
#     return loss

# def pairwise_diffloss(y_hat,y_true,sol ,solpool,margin=0, minimize=False,mode= 'B'):
#     '''
#     sol, y and y_hat are torch array [batch_size,48]
#     solpool is torch array [currentpoolsize,48]
#     '''
#     mm = 1 if minimize else -1 
#     loss = 0
#     for ii in range(len(y_true)):
#         _,indices= np.unique((mm*y_true[ii]*solpool).sum(dim=1).detach().numpy(),return_index=True)
#         ## return indices after sorting the array in ascending order
#         if mode == 'B':
#             big_ind = [indices[0] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
#         if mode == 'W':
#             big_ind = [indices[p] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
#         if mode == 'S':
#             big_ind = [indices[p] for p in range(len(indices)-1)] #good one
#             small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one

#         loss += (mm*( torch.matmul(solpool[big_ind], y_hat[ii]) - torch.matmul(solpool[small_ind], y_hat[ii]) 
#     - (torch.matmul(solpool[big_ind], y_true[ii]) - torch.matmul(solpool[small_ind], y_true[ii])) )).square().mean()
        
#     return loss


# def Listnet_loss(y_hat,y_true,sol,solpool,minimize=False,*wd,**kwd):
#     mm = 1 if minimize else -1 
#     loss = 0
#     for ii in range(len(y_true)):
#          loss += -(F.log_softmax((-mm*y_hat[ii]*solpool).sum(dim=1),
#                 dim=0)*F.softmax((-mm*y_true[ii]*solpool).sum(dim=1),dim=0)).mean()
#     return loss

# class SemanticPO(baseline_mse):
#     def __init__(self,loss_fn, solpool,growpool_fn, growth =0.0, lr=1e-1,margin=0.):
#         super().__init__(lr)
#         # self.save_hyperparameters()
#         self.loss_fn = loss_fn
#         self.solpool = solpool
#         self.growpool_fn = growpool_fn
#         self.growth = growth
#         self.margin = margin
#         self.save_hyperparameters("lr","growth","margin")
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("SemanticPO")
#         parser.add_argument("--lr", type=float, default=0.1)
#         return parent_parser  
#     def training_step(self, batch, batch_idx):
#         x,y,sol,m = batch
#         y_hat =  self(x).squeeze()
#         if (np.random.random(1)[0]< self.growth) or len(self.solpool)==0:
#             self.solpool = self.growpool_fn(self.solpool, y_hat,m)

#         loss = self.loss_fn(y_hat,y,sol,self.solpool,margin = self.margin)
#         self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
#         return loss

# class CombinedPO(baseline_mse):
#     def __init__(self,loss_fn, solpool,growpool_fn, growth =0.0, lr=1e-1,margin=0.,alpha=0.5):
#         super().__init__(lr)
#         # self.save_hyperparameters()
#         self.loss_fn = loss_fn
#         self.solpool = solpool
#         self.growpool_fn = growpool_fn
#         self.growth = growth
#         self.margin = margin
#         self.alpha = alpha
#         self.save_hyperparameters("lr","growth","margin","alpha")
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("SemanticPO")
#         parser.add_argument("--lr", type=float, default=0.1)
#         return parent_parser  
#     def training_step(self, batch, batch_idx):
#         x,y,sol,m = batch
#         y_hat =  self(x).squeeze()
#         if (np.random.random(1)[0]< self.growth) or len(self.solpool)==0:
#             self.solpool = self.growpool_fn(self.solpool, y_hat,m)

#         criterion = nn.MSELoss(reduction='mean')
#         loss = self.alpha*self.loss_fn(y_hat,y,sol,self.solpool,margin =self.margin) + (1- self.alpha)*criterion(y_hat,y)
#         self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
#         return loss