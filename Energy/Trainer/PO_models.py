import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from Trainer.comb_solver import SolveICON,  MakeLpMat
from Trainer.utils import regret_aslist, regret_fn, abs_regret_fn, growpool_fn, batch_solve



class twostage_regression(pl.LightningModule):
    def __init__(self,param, lr=1e-1, max_epochs=30, seed=0, scheduler=False, relax=False, **kwd):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            param: the parameter of the scheduling problem
            lr: learning rate
            max_epochs: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net = nn.Linear(8,1)
        self.param = param
        self.lr = lr
        self.max_epochs= max_epochs
        self.scheduler = scheduler
        self.solver = SolveICON(relax=relax, **param)
        self.solver.make_model()

    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

    def predict_step(self, batch, batch_idx):
        '''
        I am using the the predict module to compute regret !
        '''
        solver = self.solver
        
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        regret_tensor = regret_aslist(solver,y_hat,y,sol)
        return regret_tensor

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y, sol)
        abs_val_loss= abs_regret_fn(self.solver, y_hat,y, sol)
        mseloss = criterion(y_hat, y)
        self.log("val_regret", val_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_val_loss, prog_bar=False, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss}

    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(self.solver, y_hat,y, sol)
        abs_val_loss= abs_regret_fn(self.solver, y_hat,y, sol)
        mseloss = criterion(y_hat, y)
        self.log("test_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_abs_regret", abs_val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"test_regret": val_loss, "test_mse": mseloss}
    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###

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

from Trainer.diff_layer import SPOlayer, DBBlayer
class SPO(twostage_regression):
    def __init__(self,param, lr=1e-1, max_epochs=30, seed=20, scheduler=False, relax=False, **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax)
        self.layer  = SPOlayer(self.solver)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = self.layer(y_hat, y, sol)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class DBB(twostage_regression):
    def __init__(self,param,lambda_val=1., lr=1e-1, max_epochs=30, seed=20, scheduler=False, relax=False, **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax)
        self.layer  = DBBlayer(self.solver, lambda_val= lambda_val)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(y_hat, y, sol)
        loss = ((sol_hat - sol)*y).sum(-1).mean() ## to minimze regret
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
class IMLE(twostage_regression):
    def __init__(self, param, lr=0.1, max_epochs=30, seed=0, scheduler=False, relax=False,
                k=5,nb_iterations=100, nb_samples=1, beta=10.,
                temperature=1.0,  **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax, **kwd)
        target_distribution = TargetDistribution(alpha=1.0, beta= beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= k, nb_iterations=nb_iterations)
        imle_solver = lambda y_: batch_solve(self.solver, -y_)

        
        self.layer = imle( imle_solver,target_distribution=target_distribution,
        noise_distribution=noise_distribution, input_noise_temperature= temperature, 
        target_noise_temperature= temperature,nb_samples=nb_samples)

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(-y_hat)
        loss = ((sol_hat - sol)*y).sum(-1).mean()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss


from DPO import perturbations
from DPO import fenchel_young as fy
class FenchelYoung(twostage_regression):
    def __init__(self, param, lr=0.1, max_epochs=30, seed=0, scheduler=False, relax=False, num_samples=10, sigma=0.1, **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax, **kwd)
        fy_solver = lambda y_: batch_solve(self.solver, y_)
        self.criterion = fy.FenchelYoungLoss( fy_solver, num_samples= num_samples, sigma= sigma, maximize = False,
         batched=True)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        criterion  = self.criterion
        y_hat =  self(x).squeeze()
        loss = criterion(y_hat, sol).mean()

        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

from intopt.intopt import intopt
class IntOpt(twostage_regression):
    def __init__(self,param, lr=1e-1, max_epochs=30, seed=20, scheduler=False, relax=False,
        thr= 1e-8, damping= 1e-5, diffKKT = False, dopresolve = True, **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax)
        self.thr, self.damping, self.diffKKT, self.dopresolve = thr, damping, diffKKT, dopresolve
        A,b,G,h,T = MakeLpMat(**param)
        self.diff_layer = intopt(A ,b ,G ,h , thr, damping, diffKKT, dopresolve)
        self.A_trch = A.float()
        self.b_trch = b.float()
        self.G_trch = G.float()
        self.h_trch = h.float()
        self.T_trch = T.float()

    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x)

        c_hat = torch.matmul(self.T_trch, y_hat).squeeze()
        c_true = torch.matmul(self.T_trch, y.unsqueeze(2)).squeeze()
        sol_hat = self.diff_layer(c_hat)
        loss = (sol_hat * c_true).sum()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
class DCOL(twostage_regression):
    '''
    Implementation oF QPTL using cvxpyayers
    '''
    def __init__(self,param, lr=1e-1, max_epochs=30, seed=20, scheduler=False, relax=False, mu=0.1,regularizer='quadratic', **kwd):
        super().__init__(param, lr, max_epochs, seed, scheduler, relax)
        A,b,G,h,T = MakeLpMat(**param)
        # self.A_trch = A.float()
        # self.b_trch = b.float()
        # self.G_trch = G.float()
        # self.h_trch = h.float()
        self.T_trch = T.float()
        n = A.shape[1]
        c = cp.Parameter(n)
        x = cp.Variable(n)
        constraints = [x >= 0,A @ x == b,G @ x <= h ]        
        
        if regularizer=='quadratic':
            objective = cp.Minimize(c @ x+ mu*cp.pnorm(x, p=2))  
        elif regularizer=='entropic':
            objective = cp.Minimize(c @ x -  mu*cp.sum(cp.entr(x)) )
        problem = cp.Problem(objective, constraints)
        self.diff_layer = CvxpyLayer(problem, parameters=[c], variables=[x])



    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x)

        c_hat = torch.matmul(self.T_trch, y_hat).squeeze()
        c_true = torch.matmul(self.T_trch, y.unsqueeze(2)).squeeze()
        sol_hat, = self.diff_layer(c_hat)

        loss = (sol_hat * c_true).sum()
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss


from Trainer.CacheLosses import *
class CachingPO(twostage_regression):
    def __init__(self,loss,param,init_cache, growth =0.1, lr=1e-1,tau=0.,
        max_epochs=30, seed=20, scheduler=False, relax=False, **kwd):
        '''
        tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''
        super().__init__(param, lr, max_epochs, seed, scheduler, relax)
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
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = growpool_fn(self.solver, self.cache, y_hat)

        loss = self.loss_fn(y_hat,y,sol,self.cache)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class CombinedPO(CachingPO):
    def __init__(self,alpha, loss,param,init_cache, growth =0.1, lr=1e-1,tau=0.,
        max_epochs=30, seed=20, scheduler=False, relax=False, **kwd):
        '''
        tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''
        super().__init__(loss,param,init_cache, growth , lr,tau, max_epochs, seed, scheduler, relax)
        self.alpha = alpha
        self.save_hyperparameters("lr","growth","tau","alpha")
    
 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = growpool_fn(self.cache, y_hat, self.param)
        criterion = nn.MSELoss(reduction='mean')
        loss = self.alpha* self.loss_fn(y_hat,y,sol,self.cache,tau=self.tau) + (1 - self.alpha)*criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss