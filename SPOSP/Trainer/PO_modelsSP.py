import logging
import torch 
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from Trainer.losses import *
from Trainer.utils import  regret_fn, regret_aslist, growcache
from Trainer.optimizer_module import spsolver, cvxsolver, intoptsolver, qpsolver
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
from DPO import perturbations
from DPO import fenchel_young as fy
logging.basicConfig(filename='Uniquesolutions.log', level=logging.INFO)

class twostage_regression(pl.LightningModule):
    def __init__(self,net,exact_solver = spsolver, lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
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
        self.max_epochs= max_epochs
        self.save_hyperparameters("lr",'l1_weight')
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
        # pointwise_loss = pointwise_crossproduct_loss(y_hat,y)

        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
        # self.log("val_pointwise", pointwise_loss, prog_bar=True, on_step=True, on_epoch=True, )

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
        # pointwise_loss = pointwise_crossproduct_loss(y_hat,y)

        self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
        # self.log("test_pointwise", pointwise_loss, prog_bar=True, on_step=True, on_epoch=True, )

        return {"test_mse":mseloss, "test_regret":regret_loss}
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     num_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, epochs=self.max_epochs,
    #     steps_per_epoch = num_batches)
    #     return [optimizer], [scheduler]
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

class SPO(twostage_regression):
    def __init__(self,net,exact_solver = spsolver,loss_fn=SPOLoss,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        """
        Implementaion of SPO+ loss subclass of twostage model
            loss_fn: loss function 

 
        """
        super().__init__(net,exact_solver, lr, l1_weight,max_epochs, seed)
        self.loss_fn = loss_fn
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        for ii in range(len(y)):
            loss += self.loss_fn(self.exact_solver)(y_hat[ii],y[ii], sol[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  

class Blackbox(twostage_regression):
    """
    Implemenation of Blackbox differentiation gradient
    """
    def __init__(self,net,exact_solver = spsolver,lr=1e-1,mu =0.1, l1_weight=0.1,max_epochs=30, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)
        self.mu = mu
        self.save_hyperparameters("lr","mu")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            loss += BlackboxLoss(self.exact_solver,self.mu)(y_hat[ii],y[ii], sol[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss   

class CachingPO(twostage_regression):
    def __init__(self,loss_fn,init_cache, net,exact_solver = spsolver,growth=0.1,tau=0.,lr=1e-1,
        l1_weight=0.1,max_epochs=30, seed=20):
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
        super().__init__(net,exact_solver, lr, l1_weight,max_epochs, seed)
        # self.save_hyperparameters()
        self.loss_fn = loss_fn
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

  
        loss = self.loss_fn(y_hat,y,sol,self.cache, self.tau)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  



###################################### This approach use it's own solver #########################################



class DCOL(twostage_regression):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20,mu=0.1):
        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)
        self.solver = cvxsolver(mu=mu)
    def training_step(self, batch, batch_idx):
        solver = self.solver
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            sol_hat = solver.shortest_pathsolution(y_hat[ii])
            ### The loss is regret but c.dot(y) is constant so need not to be considered
            loss +=  (sol_hat ).dot(y[ii])
 
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss 

class QPTL(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1,  max_epochs=30, seed=20,mu=0.1):
        

        super().__init__(net,exact_solver,lr, l1_weight,max_epochs, seed, mu)  
        self.solver = qpsolver( mu=mu)
    
class IntOpt(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,thr=0.1,damping=1e-3,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        

        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)  
        self.solver  = intoptsolver(thr=thr,damping=damping)



###################################### I-MLE #########################################
########## Code adapted from https://github.com/uclnlp/torch-imle/blob/main/annotation-cli.py ###########################



class IMLE(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,k=5,nb_iterations=100,nb_samples=1, 
            input_noise_temperature=1.0, target_noise_temperature=1.0,lr=1e-1,l1_weight=0.1,max_epochs=30,seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, max_epochs, seed)
        self.solver = solver
        self.k = k
        self.nb_iterations = nb_iterations
        self.nb_samples = nb_samples
        self.target_noise_temperature = target_noise_temperature
        self.input_noise_temperature = input_noise_temperature

        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0

        # input_noise_temperature = 1.0
        # target_noise_temperature = 1.0

        target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
        noise_distribution = SumOfGammaNoiseDistribution(k= self.k, nb_iterations=self.nb_iterations)

        @imle(target_distribution=target_distribution,
                noise_distribution=noise_distribution,
                input_noise_temperature=self.input_noise_temperature,
                target_noise_temperature=self.target_noise_temperature,
                nb_samples=self.nb_samples)
        def imle_solver(y):
            #     I-MLE assumes that the solver solves a maximisation problem, but here the `solver` function solves
            # a minimisation problem, so we flip the sign twice. Feed negative cost coefficient to imle_solver and then 
            # flip it again to feed the actual cost to the solver
            return spsolver.solution_fromtorch(-y)

        ########### Also the forward pass returns the solution of the perturbed cost, which is bit strange
        ###########
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        for ii in range(len(y)):
            sol_hat = imle_solver(-y_hat[ii].unsqueeze(0)) # Feed neagtive cost coefficient
            loss +=  (sol_hat*y[ii]).mean()

        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss 
###################################### Differentiable Perturbed Optimizer #########################################

class DPO(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,lr=1e-1,l1_weight=0.1, max_epochs= 30, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, max_epochs, seed)
        self.solver = solver
        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        @perturbations.perturbed(num_samples=10, sigma=0.1, noise='gumbel',batched = False)
        def dpo_solver(y):
            return spsolver.solution_fromtorch(-y)

        for ii in range(len(y)):
            sol_hat = dpo_solver(-y_hat[ii]) # Feed neagtive cost coefficient
            loss +=  ( sol_hat  ).dot(y[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss




################################ Implementation of a Fenchel-Young loss using perturbation techniques #########################################

class FenchelYoung(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,num_samples=10, sigma=0.1,lr=1e-1, l1_weight=1e-5, max_epochs=30, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, max_epochs, seed)
        self.solver = solver
        self.num_samples = num_samples
        self.sigma = sigma
        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        solver= self.solver
        

        
        def fy_solver(y):
            return spsolver.solution_fromtorch(y)
        ############# Define the Loss functions, we can set maximization to be false

        criterion = fy.FenchelYoungLoss(fy_solver, num_samples= self.num_samples, sigma= self.sigma,maximize = False, batched=False)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            sol_true = fy_solver(y[ii])
            loss +=  criterion(y_hat[ii],sol_true)

        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss 
