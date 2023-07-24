import logging
import torch 
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from Trainer.diff_layer import *
from Trainer.utils import  regret_fn, regret_list, abs_regret_fn, growcache
# from Trainer.optimizer_module import spsolver, cvxsolver,  qpsolver, intoptsolver
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
from DPO import perturbations
from DPO import fenchel_young as fy

class baseline(pl.LightningModule):
    def __init__(self,net,exact_solver ,cov, gamma, lr=1e-1, l1_weight=1e-5,max_epochs=30, seed=20, scheduler=False, **kwd):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net =  net
        self.lr = lr
        self.l1_weight = l1_weight
        self.exact_solver = exact_solver
        self.max_epochs = max_epochs
        self.scheduler = scheduler
    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        
        x,y, sol = batch
        
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss =  loss  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss,  on_step=True, on_epoch=True, )
        return training_loss 
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        
        y_hat =  self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_loss =  regret_fn(self.exact_solver, y_hat,y, sol) 
        abs_regret_loss =  abs_regret_fn(self.exact_solver, y_hat,y, sol) 
        abs_pred = torch.abs( y_hat ).mean()



        self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_abs_regret", abs_regret_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_absolute_value", abs_pred, prog_bar=True, on_step=False, on_epoch=True, )
        # self.log("val_pointwise", pointwise_loss, prog_bar=True, on_step=False, on_epoch=True, )

        return {"val_mse":mseloss, "val_regret":regret_loss}
    def validation_epoch_end(self, outputs):
        avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
        self.log("ptl/val_regret", avg_regret)
        self.log("ptl/val_mse", avg_mse)
        # self.log("ptl/val_accuracy", avg_acc)
        
    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_loss =  regret_fn(self.exact_solver, y_hat,y, sol) 
        abs_regret_loss =  abs_regret_fn(self.exact_solver, y_hat,y, sol) 
        abs_pred = torch.abs( y_hat ).mean()


        self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_regret", regret_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("test_abs_regret", abs_regret_loss, prog_bar=True, on_step=False, on_epoch=True, )
        self.log("val_absolute_value", abs_pred, prog_bar=True, on_step=False, on_epoch=True, )
        # self.log("test_pointwise", pointwise_loss, prog_bar=True, on_step=True, on_epoch=True, )

        return {"test_mse":mseloss, "test_regret":regret_loss}

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        # return [self.opt], [self.reduce_lr_on_plateau]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.2,
                patience=2,
                min_lr=1e-6),
                        "monitor": "val_regret",
                        # "frequency": "indicates how often the metric is updated"
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }
        return optimizer

class SPO(baseline):
    def __init__(self,net,exact_solver,cov, gamma,lr=1e-1, l1_weight=1e-5,max_epochs=30, seed=20, scheduler=False, **kwd):
        """
        Implementaion of SPO+ loss subclass of twostage model
            loss_fn: loss function 

 
        """
        super().__init__(net,exact_solver, cov, gamma, lr, l1_weight,max_epochs, seed, scheduler)
        self.loss_fn =  SPOlayer(self.exact_solver)

    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        # for ii in range(len(y)):
        #     loss += self.loss_fn(y_hat[ii],y[ii], sol[ii])
        training_loss = self.loss_fn(y_hat,y, sol)/len(y) + l1penalty * self.l1_weight
        # training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  

class DBB(baseline):
    """
    Implemenation of Blackbox differentiation gradient
    """
    def __init__(self,net,exact_solver, cov, gamma,lr=1e-1,lambda_val =0.1, l1_weight=1e-5,max_epochs=30, seed=20, scheduler=False, **kwd):
        super().__init__(net,exact_solver, cov, gamma, lr, l1_weight,max_epochs, seed, scheduler)
        self.lambda_val = lambda_val
        self.layer = DBBlayer(self.exact_solver,self.lambda_val)
        self.save_hyperparameters("lr","lambda_val")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        sol_hat = self.layer(y_hat, y, sol)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        training_loss =  ((sol_hat - sol)*y).sum(-1).mean() + l1penalty * self.l1_weight

        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss", ((sol_hat - sol)*y).sum(-1).mean(),  on_step=True, on_epoch=True, )
        return training_loss   
from Trainer.CacheLosses import *
class CachingPO(baseline):
    def __init__(self,loss,init_cache, net,exact_solver,  cov, gamma, growth=0.1,tau=0.,lr=1e-1,
        l1_weight=1e-5,max_epochs=30, seed=20, scheduler=False, **kwd):
        """
        A class to implement loss functions using soluton cache
        Args:
            loss_fn: the loss function (NCE, MAP or the rank-based ones)
            init_cache: initial solution cache
            growth: p_solve
            tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 

        """
        super().__init__(net,exact_solver, cov, gamma, lr, l1_weight,max_epochs, seed, scheduler)
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
        ### The cache
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        self.cache = torch.from_numpy(init_cache_np).float()
        self.growth = growth
        self.tau = tau
        self.save_hyperparameters("lr","growth","tau")
    
 
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]<= self.growth) or len(self.cache)==0:
            self.cache = growcache(self.exact_solver, self.cache, y_hat)

  
        loss = self.loss_fn(y_hat,y,sol,self.cache)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  



###################################### This approach use it's own solver #########################################


from Trainer.optimizer_module import cvxsolver
class DCOL(baseline):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver,cov, gamma, lr=1e-1, l1_weight=1e-5,max_epochs=30, seed=20,mu=0.1,regularizer='quadratic', scheduler= False,**kwd):
        super().__init__(net,exact_solver, cov, gamma, lr, l1_weight,max_epochs, seed, scheduler)
        self.layer = cvxsolver( cov=cov, gamma=gamma,  mu=mu, regularizer=regularizer)
    def training_step(self, batch, batch_idx):
   
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        sol_hat = self.layer.solution(y_hat)
        training_loss =  ((sol_hat - sol)*y).sum(-1).mean() + l1penalty * self.l1_weight

        # for ii in range(len(y)):
        #     sol_hat = self.layer.shortest_pathsolution(y_hat[ii])
        #     ### The loss is regret but c.dot(y) is constant so need not to be considered
        #     loss +=  (sol_hat ).dot(y[ii])
 
        # training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss 


##################################### I-MLE #########################################
######### Code adapted from https://github.com/uclnlp/torch-imle/blob/main/annotation-cli.py ###########################
 


class IMLE(baseline):
    def __init__(self,net,exact_solver,cov, gamma,   k=5,nb_iterations=100,nb_samples=1, beta=10.,
            temperature=1.0, lr=1e-1,l1_weight=1e-5,max_epochs=30,seed=20 , scheduler=False, **kwd):
        super().__init__(net,exact_solver , cov, gamma,lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = exact_solver
        self.k = k
        self.nb_iterations = nb_iterations
        self.nb_samples = nb_samples
        # self.target_noise_temperature = target_noise_temperature
        # self.input_noise_temperature = input_noise_temperature
        target_distribution = TargetDistribution(alpha=1.0, beta= beta)
        noise_distribution = SumOfGammaNoiseDistribution(k= self.k, nb_iterations=self.nb_iterations)

        imle_solver = lambda y_: self.solver.solution_fromtorch(-y_)

        self.imle_layer = imle(imle_solver,target_distribution=target_distribution,
        noise_distribution=noise_distribution, input_noise_temperature= temperature, 
        target_noise_temperature= temperature,nb_samples=self.nb_samples)
        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        
        sol_hat = self.imle_layer(-y_hat)
        loss = ((sol_hat - sol)*y).sum(-1).mean()
        training_loss= loss  + l1penalty * self.l1_weight

        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss,  on_step=True, on_epoch=True, )
        return training_loss 
###################################### Differentiable Perturbed Optimizer #########################################

class DPO(baseline):
    def __init__(self,net,exact_solver,cov, gamma, num_samples=10, sigma=0.1, lr=1e-1,l1_weight=1e-5, max_epochs= 30, seed=20, scheduler=False, **kwd):
        super().__init__(net,exact_solver ,cov, gamma, lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = exact_solver
        @perturbations.perturbed(num_samples= num_samples, sigma= sigma, noise='gumbel',batched = True)
        def dpo_layer(y):
            return exact_solver.solution_fromtorch(y)
        self.dpo_layer = dpo_layer


        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        
        sol_hat = self.dpo_layer(y_hat)
        loss = ((sol_hat - sol)*y).sum(-1).mean()
        training_loss= loss  + l1penalty * self.l1_weight        
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss,  on_step=True, on_epoch=True, )
        return training_loss




################################ Implementation of a Fenchel-Young loss using perturbation techniques #########################################

class FenchelYoung(baseline):
    def __init__(self,net,exact_solver, cov, gamma,num_samples=10, sigma=0.1,lr=1e-1, l1_weight=1e-5, max_epochs=30, seed=20, scheduler=False, **kwd):
        super().__init__(net,exact_solver , cov, gamma,lr, l1_weight, max_epochs, seed, scheduler)
        self.solver = exact_solver
        self.num_samples = num_samples
        self.sigma = sigma
        self.save_hyperparameters("lr")
        self.fy_solver = lambda y_: exact_solver.solution_fromtorch(y_)
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
     

        criterion = fy.FenchelYoungLoss(self.fy_solver, num_samples= self.num_samples, sigma= self.sigma,maximize = False,
         batched=True)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        loss = criterion(y_hat, sol).mean()
    

        training_loss=  loss + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss,  on_step=True, on_epoch=True, )
        return training_loss 
