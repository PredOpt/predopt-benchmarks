
from Trainer.NNModels import cora_net, cora_normednet, cora_nosigmoidnet
from Trainer.utils import regret_fn, regret_list, growpool_fn, abs_regret_fn
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
    def __init__(self,solver,lr=1e-1,mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__()
        pl.seed_everything(seed)
        if mode=='sigmoid':
            self.model = cora_net(n_layers= n_layers)
            
        elif mode=="batchnorm" :
            self.model = cora_normednet(n_layers= n_layers)
        elif mode=="linear":
            self.model = cora_nosigmoidnet(n_layers= n_layers)
        self.mode= mode

        self.lr = lr
        self.solver = solver
        self.scheduler = scheduler
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
        abs_val_loss= abs_regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)

        if self.mode!= "sigmoid":
           y_hat = torch.sigmoid(y_hat)
        criterion2 = nn.BCELoss(reduction='mean')
        bceloss = criterion2(y_hat, sol)


        self.log("val_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_bce", bceloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, ) 
       
        return  {"val_regret": val_loss, "val_mse": mseloss}

    def test_step(self, batch, batch_idx):
        solver = self.solver
        
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(solver,y_hat,y,sol,m)
        abs_val_loss= abs_regret_fn(solver,y_hat,y,sol,m)
        criterion1 = nn.MSELoss(reduction='mean')
        mseloss = criterion1(y_hat, y)
        criterion2 = nn.BCELoss(reduction='mean')
        if self.mode!= "sigmoid":
           y_hat = torch.sigmoid(y_hat)

        bceloss = criterion2(y_hat, sol)
        self.log("test_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )    
        self.log("test_bce", bceloss, prog_bar=True, on_step=False, on_epoch=True, )  
        self.log("test_abs_regret", abs_val_loss, prog_bar=True, on_step=False, on_epoch=True, )        
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
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6),
                    "monitor": "val_regret"
            }
            }
        return optimizer

class baseline_bce(baseline_mse):
    def __init__(self,solver,lr=1e-1,mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
            super().__init__(solver,lr,mode,n_layers,seed, scheduler) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(y_hat,sol)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss   


class SPO(baseline_mse):
    def __init__(self,solver, lr=1e-1,mode='sigmoid',n_layers=2,seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = SPOlayer(solver)
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss =  self.layer(y_hat, y,sol,m ) 
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB(baseline_mse):
    def __init__(self, solver,lr=1e-1,lambda_val=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.layer = DBBlayer(solver,lambda_val=lambda_val)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        sol_hat  =  self.layer(y_hat, y,sol,m ) 
        loss = ((sol - sol_hat)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class FenchelYoung(baseline_mse):
    def __init__(self,solver,sigma=0.1,num_samples=10, 
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        # fy_solver = lambda y_: batch_solve(self.solver,y_,m)
        loss = 0
        for i in range(len(y_hat)):
            def fy_solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()

            # fy_solver = lambda y_: batch_solve(self.solver,y_,m[i],batched=False)
            criterion = fy.FenchelYoungLoss(fy_solver, num_samples= self.num_samples, sigma= self.sigma,maximize = True, batched= False)
            loss += criterion(y_hat[i], sol[i]).mean()
        # criterion = fy.FenchelYoungLoss(fy_solver, num_samples= self.num_samples, sigma= self.sigma,maximize = True, batched= True)
        # loss = criterion(y_hat, sol).mean()
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DPO(baseline_mse):
    def __init__(self,solver,sigma=0.1,num_samples=10, 
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        self.sigma = sigma
        self.num_samples = num_samples
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = 0
        for i in range(len(y_hat)):
            def solver(y_):
                sol = []
                ### FY extend the size of y to num_sample*batch
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                
                return torch.cat(sol).float()
            op = perturbations.perturbed(solver, num_samples= self.num_samples, sigma= self.sigma,noise='gumbel', batched= False)( y_hat[i] )
            loss += y[i].dot(sol[i] - op)
        loss /= len(y_hat)

        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss



class IMLE(baseline_mse):
    def __init__(self,solver,k=5, nb_iterations=100,nb_samples=1, beta=10.0,
            temperature=1.0,
            lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):

        super().__init__(solver,lr,mode,n_layers,seed, scheduler) 

        self.target_distribution = TargetDistribution(alpha=1.0, beta=beta)
        self.noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations= nb_iterations)

        self.input_noise_temperature= temperature
        self.target_noise_temperature= temperature
        self.nb_samples= nb_samples
    def training_step(self, batch, batch_idx):

        input_noise_temperature= self.input_noise_temperature
        target_noise_temperature= self.target_noise_temperature
        nb_samples= self.nb_samples
        target_distribution = self.target_distribution
        noise_distribution = self.noise_distribution


        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = 0
        for i in range(len(y_hat)):
            def imle_solver(y_):
                sol = []
                for j in range(len(y_)):
                     sol.append(  batch_solve(self.solver,y_[j],m[i],batched=False).unsqueeze(0) )
                return torch.cat(sol).float()
            op = imle(imle_solver,  target_distribution=target_distribution,noise_distribution=noise_distribution,
                    input_noise_temperature= input_noise_temperature, target_noise_temperature= target_noise_temperature,
                    nb_samples= nb_samples)( y_hat[i].view(1,-1) ).squeeze()
            loss += y[i].dot(sol[i] - op)
        loss /= len(y_hat)

        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
from qpth.qp import QPFunction
class QPTL(baseline_mse):
    def __init__(self, solver,lr=1e-1,mu=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.mu = mu
    def training_step(self, batch, batch_idx):
        mu = self.mu
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()

        loss = 0
        for i in range(len(y_hat)):
            A,b, G,h  = self.solver.get_qpt_matrices(m[i])
            Q =  mu*torch.eye(G.shape[1]).float()
            op = QPFunction()(Q,-y_hat[i],G,h,A,b).squeeze()
    
            loss +=  y[i].dot(sol[i] - op)
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

import cvxpy as cp  
from cvxpylayers.torch import CvxpyLayer
class DCOL(baseline_mse):
    def __init__(self, solver,lr=1e-1,mu=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.mu = mu
    def training_step(self, batch, batch_idx):
        mu = self.mu
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()

        loss = 0
        for i in range(len(y_hat)):
            A,b, G,h  = self.solver.get_qpt_matrices(m[i])
            z = cp.Variable(G.shape[1])
            c = cp.Parameter(G.shape[1])
            constraints = [G @ z <=h]  
            objective = cp.Maximize(c @ z - mu*cp.pnorm(z, p=2))  #cp.pnorm(A @ x - b, p=1)
            problem = cp.Problem(objective, constraints)
            op,  = CvxpyLayer(problem, parameters=[c], variables=[z])(y_hat[i])
            loss +=  y[i].dot(sol[i] - op)
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss            
from intopt.intopt import intopt_nonbacthed
class IntOpt(baseline_mse):
    def __init__(self, solver,lr=1e-1,mu=0.1,mode='sigmoid',n_layers=2, seed=0,scheduler=False,
        thr= 1e-8, damping= 1e-5, diffKKT = False, dopresolve = True, **kwd):
        super().__init__(solver,lr,mode,n_layers,seed, scheduler)
        self.thr, self.damping, self.diffKKT, self.dopresolve = thr, damping, diffKKT, dopresolve

    def training_step(self, batch, batch_idx):
        thr, damping, diffKKT, dopresolve = self.thr, self.damping, self.diffKKT, self.dopresolve
        x,y,sol,m = batch
        y_hat =  self(x).squeeze() 
        loss = 0
        for i in range(len(y_hat)):
            A,b, G,h  = self.solver.get_qpt_matrices(m[i])

            op =  intopt_nonbacthed(A ,b ,G ,h , thr, damping, diffKKT, dopresolve)(-y_hat[i])
            loss +=  y[i].dot(sol[i] - op)
        loss /= len(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
from Trainer.CacheLosses import *
class CachingPO(baseline_mse):
    def __init__(self,solver,init_cache,tau=1.,growth=0.1,loss="listwise",
        lr=1e-1,mode='sigmoid',n_layers=2, seed=0,scheduler=False, **kwd):
        '''tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''
        super().__init__(solver,lr,mode,n_layers,seed, scheduler) 
        # self.save_hyperparameters()
        if loss=="pointwise":
            self.loss_fn = PointwiseLoss()
        elif loss=="pairwise":
            self.loss_fn = PairwiseLoss(margin=tau)
        elif loss == "pairwise_diff":
            self.loss_fn = PairwisediffLoss()
        elif loss == "listwise":
            self.loss_fn = ListwiseLoss(temperature=tau)
        elif loss== 'NCE':
            self.loss_fn = NCE()
        elif loss== 'MAP':
            self.loss_fn = MAP()
        elif loss== 'NCE_c':
            self.loss_fn = NCE_c()
        elif loss== 'MAP_c':
            self.loss_fn = MAP_c()
        elif loss== 'MAP_c_actual':
            self.loss_fn = MAP_c_actual()
        else:
            raise Exception("Invalid Loss Provided")
        self.growth = growth
        cache_np = init_cache.detach().numpy()
        cache_np = np.unique(cache_np,axis=0)
        # torch has no unique function, so we have to do this
        init_cache =  torch.from_numpy(cache_np).float()
        self.cache = init_cache

    

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache= growpool_fn(self.solver,self.cache, y_hat,m)

        loss = self.loss_fn(y_hat,y,sol,self.cache)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss 

