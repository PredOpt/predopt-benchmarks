import os
import logging
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import networkx as nx
import gurobipy as gp

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
from DPO import perturbations
from DPO import fenchel_young as fy
logging.basicConfig(filename='Uniquesolutions.log', level=logging.INFO)
###################################### Graph Structure ###################################################
V = range(25)
E = []

for i in V:
    if (i+1)%5 !=0:
        E.append((i,i+1))
    if i+5<25:
            E.append((i,i+5))

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)

###################################### Gurobi Shortest path Solver #########################################
class shortestpath_solver:
    def __init__(self,G= G):
        self.G = G
    
    def shortest_pathsolution(self, y):
        '''
        y: the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status==2:
            return x.x
    def is_uniquesolution(self, y):
        '''
        y: the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 100)
        #model.PoolObjBound(obj)
        model.setParam('PoolGap', 0.0)
        model.optimize()
        self.model = model
        return model.SolCount<=1 

    def highest_regretsolution(self,y,y_true, minimize=True):
        mm = 1 if minimize else -1
        
        if self.is_uniquesolution(y):
            model = self.model
            return np.array(model.Xn).astype(np.float32), 0
        else:
            model = self.model
            sols = []
            for solindex in range(model.SolCount):
                model.setParam('SolutionNumber', solindex)
                sols.append(model.Xn)  
            sols = np.array(sols).astype(np.float32)
            # print(sols.dot(y_true))
            return sols[np.argmax(sols.dot(y_true)*mm, axis=0)], 1 


    def solution_fromtorch(self,y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.shortest_pathsolution( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)
    def highest_regretsolution_fromtorch(self,y_hat,y_true,minimize=True):
        if y_hat.dim()==1:
            sol, nonunique_cnt = self.highest_regretsolution( y_hat.detach().numpy(),
                     y_true.detach().numpy(),minimize  )
            return torch.from_numpy(sol).float(), nonunique_cnt
        else:
            solutions = []
            nonunique_cnt =0
            for ii in range(len(y_hat)):
                sol,nn = self.highest_regretsolution( y_hat[ii].detach().numpy(),
                     y_true[ii].detach().numpy(),minimize )
                solutions.append(torch.from_numpy(sol).float())
                nonunique_cnt += nn
            return torch.stack(solutions) , nonunique_cnt      
        
spsolver =  shortestpath_solver()
###################################### Wrapper #########################################
class datawrapper():
    def __init__(self, x,y, sol=None, solver= spsolver ):
        self.x = x
        self.y = y
        if sol is None:
            sol = []
            for i in range(len(y)):
                sol.append(   solver.shortest_pathsolution(y[i])   )            
            sol = np.array(sol).astype(np.float32)
        self.sol = sol

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.sol[index]


###################################### Dataloader #########################################

class ShortestPathDataModule(pl.LightningDataModule):
    def __init__(self, train_df,valid_df,test_df,generator, batchsize: int = 32, num_workers: int=8):
        super().__init__()
        self.train_df = train_df
        self.valid_df =  valid_df
        self.test_df = test_df
        self.batchsize = batchsize
        self.generator =  generator
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batchsize,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batchsize,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=1000, num_workers=self.num_workers)


def batch_solve(solver, y,relaxation =False):
    sol = []
    for i in range(len(y)):
        sol.append(   solver.solution_fromtorch(y[i]).reshape(1,-1)   )
    return torch.cat(sol,0)


def regret_fn(solver, y_hat,y_true, sol_true, minimize= True):  
    '''
    computes regret given predicted y_hat and true y
    '''
    regret_list = []
    for ii in range(len(y_true)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii], sol_true[ii]) )
    return torch.mean( torch.tensor(regret_list ))

def regret_aslist(solver, y_hat,y_true, sol_true, minimize= True):  
    '''
    computes regret of more than one cost vectors
    ''' 
    regret_list = []
    for ii in range(len(y_true)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii], sol_true[ii]).item() )
    return np.array(regret_list)

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

    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i,
    #                    second_order_closure=None):
    #     self.opt.step()
    #     self.opt.zero_grad()
    #     if self.trainer.global_step % self.val_check_interval == 0:
    #         self.reduce_lr_on_plateau.step(self.current_val_loss)
###################################### SPO and Blackbox #########################################

def SPOLoss(solver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true, sol_true):
       
            sol_hat = solver.solution_fromtorch(y_pred)
            sol_spo = solver.solution_fromtorch(2* y_pred - y_true)
            # sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return mm*(sol_true - sol_spo), None, None
            
    return SPOLoss_cls.apply

def WorstcaseSPOLoss(solver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true, sol_true):
       
            # sol_hat = solver.solution_fromtorch(y_pred)
            sol_hat,  nonunique_cnt = solver.highest_regretsolution_fromtorch(y_pred,y_true,minimize=True)
            sol_spo = solver.solution_fromtorch(2* y_pred - y_true)
            # sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            logging.info("{}/{} number of y have Nonunique solutions".format(nonunique_cnt,len(y_pred)))
            return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return mm*(sol_true - sol_spo), None, None
            
    return SPOLoss_cls.apply



def BlackboxLoss(solver,mu=0.1, minimize=True):
    mm = 1 if minimize else -1
    class BlackboxLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true, sol_true):
            sol_hat = solver.solution_fromtorch(y_pred)
            sol_perturbed = solver.solution_fromtorch(y_pred + mu* y_true)
            # sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -mm*(sol_hat - sol_perturbed)/mu, None, None
            
    return BlackboxLoss_cls.apply



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


###################################### Ranking Loss  #########################################

def pointwise_mse_loss(y_hat,y_true):
    c_hat = y_hat.unsqueeze(-1)
    c_true = y_true.unsqueeze(-1)
    c_diff = c_hat - c_true
    loss = ( c_diff.square().sum())/len(c_diff)
    return loss   

def pointwise_crossproduct_loss(y_hat,y_true):
    c_hat = y_hat.unsqueeze(-1)
    c_true = y_true.unsqueeze(-1)
    c_diff = c_hat - c_true
    loss = (torch.bmm(c_diff, c_diff.transpose(2,1)).sum() )/len(c_diff)
    return loss   

def pointwise_custom_loss(y_hat,y_true, *wd,**kwd):
    loss =  pointwise_mse_loss(y_hat,y_true) + pointwise_crossproduct_loss(y_hat,y_true)
    return loss 




def pointwise_loss(y_hat,y_true,cache,*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    f(y_hat,s) is regresson on f(y,s)
    '''
    loss = (torch.matmul(y_hat,cache.transpose(0,1))- torch.matmul(y_true,cache.transpose(0,1))).square().mean()

    return loss



def pairwise_loss(y_hat,y_true,cache,tau=0, minimize=True,mode= 'B'):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    relu = nn.ReLU()
    for ii in range(len(y_true)):
        _,indices= np.unique((mm*y_true[ii]*cache).sum(dim=1).detach().numpy(),return_index=True)
        ## return indices after sorting the array in ascending order
        if mode == 'B':
            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
        if mode == 'W':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
        if mode == 'S':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one


        loss += relu(tau+ mm*( torch.matmul(cache[big_ind], y_hat[ii]) - torch.matmul(cache[small_ind], y_hat[ii])) ).mean()
        
    return loss

def pairwise_diffloss(y_hat,y_true,cache,tau=0, minimize=True,mode= 'B'):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
        _,indices= np.unique((mm*y_true[ii]*cache).sum(dim=1).detach().numpy(),return_index=True)
        ## return indices after sorting the array in ascending order
        if mode == 'B':
            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
        if mode == 'W':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
        if mode == 'S':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one

        loss += (mm*( torch.matmul(cache[big_ind], y_hat[ii]) - torch.matmul(cache[small_ind], y_hat[ii]) 
    - (torch.matmul(cache[big_ind], y_true[ii]) - torch.matmul(cache[small_ind], y_true[ii])) )).square().mean()
        
    return loss

def Listnet_loss(y_hat,y_true,cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += -(F.log_softmax((-mm*y_hat[ii]*cache/tau).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss
def Listnet_KLloss(y_hat,y_true,cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += ( F.log_softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0) -
         F.log_softmax((-mm*y_hat[ii]*cache).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss

def MAP(sol,y,cache,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.max(((sol[ii] - cache )*(mm*y[ii]  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,cache,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat 
    return MAP(sol,y,cache,minimize)
def MAP_hatc_c(y_hat,y_true,cache,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat - y_true
    return MAP(sol,y,cache,minimize)

def NCE(sol,y,cache,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.mean(((sol[ii] - cache )*(mm*y[ii]  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,cache,minimize=True,*wd,**kwd):
    sol = batch_solve( spsolver, y_hat,relaxation =False)
    y = y_hat 
    return NCE(sol,y,cache,minimize)
def NCE_hatc_c(y_hat,y_true,cache,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat - y_true
    return NCE(sol,y,cache,minimize)



def growcache(solver, cache, y_hat):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol = batch_solve(solver, y_hat,relaxation =False).detach().numpy()
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,sol,axis=0),axis=0)
    # torch has no unique function, so we need to do this
    return torch.from_numpy(cache_np).float()


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

  
        loss = self.loss_fn(y_hat,y,self.cache, self.tau)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  



###################################### This approach use it's own solver #########################################
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

### Build cvxpy modle prototype
class cvxsolver:
    def __init__(self,G=G):
        self.G = G
    def make_proto(self):
        #### Maybe we can model a better LP formulation
        G = self.G
        num_nodes, num_edges = G.number_of_nodes(),  G.number_of_edges()
        A = cp.Parameter((num_nodes, num_edges))
        b = cp.Parameter(num_nodes)
        c = cp.Parameter(num_edges)
        x = cp.Variable(num_edges)
        constraints = [x >= 0,x<=1,A @ x == b]
        objective = cp.Minimize(c @ x)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[A, b,c], variables=[x])
    def shortest_pathsolution(self, y):
        self.make_proto()
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] =1        
        sol, = self.layer(A,b,y)
        return sol
    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())         



class DCOL(twostage_regression):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)
        self.solver = cvxsolver()
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



from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model

class qpsolver:
    def __init__(self,G=G,mu=1e-6):
        self.G = G
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        self.mu = mu
        G_lb = -1*np.eye(A.shape[1])
        h_lb = np.zeros(A.shape[1])
        G_ub = np.eye(A.shape[1])
        h_ub = np.ones(A.shape[1])
        G_ineq = np.concatenate((G_lb, G_ub))
        h_ineq = np.concatenate((h_lb, h_ub))

        # G_ineq = G_lb
        # h_ineq = h_lb


        self.model_params_quad = make_gurobi_model(G_ineq,h_ineq,
            A, b, np.zeros((A.shape[1],A.shape[1]))  ) #mu*np.eye(A.shape[1])
        self.solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
                        model_params=self.model_params_quad)
    def shortest_pathsolution(self, y):
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1      
        Q =    self.mu*torch.eye(A.shape[1])
        ###########   There are two ways we can set the cosntraints of 0<= x <=1
        ########### Either specifying in matrix form, or changing the lb and ub in the qp.py file
        ########### Curretnyly We're specifying it in constraint form


        G_lb = -1*torch.eye(A.shape[1])
        h_lb = torch.zeros(A.shape[1])
        G_ub = torch.eye(A.shape[1])
        h_ub = torch.ones(A.shape[1])
        G_ineq = torch.cat((G_lb,G_ub))
        h_ineq = torch.cat((h_lb,h_ub))
        # G_ineq = G_lb
        # h_ineq = h_lb


        sol = self.solver(Q.expand(1, *Q.shape),
                            y , 
                            G_ineq.expand(1,*G_ineq.shape), h_ineq.expand(1,*h_ineq.shape), 
                            A.expand(1, *A.shape),b.expand(1, *b.shape))

        return sol.squeeze()
    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())  


class QPTL(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,mu=0.1,exact_solver = spsolver,lr=1e-1, l1_weight=0.1,  max_epochs=30, seed=20,):
        self.solver  = qpsolver(mu=mu)

        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)  
        self.solver = qpsolver( mu=mu)

from intopt.intopt_model import IPOfunc

class intoptsolver:
    def __init__(self,G=G,thr=0.1,damping=1e-3,):
        self.G = G
        self.thr =thr
        self.damping = damping
    def shortest_pathsolution(self, y):
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1      
        sol = IPOfunc(A,b,G=None,h=None,thr=self.thr,damping= self.damping)(y)
        return sol
    
class IntOpt(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,thr=0.1,damping=1e-3,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        self.solver  = intoptsolver(thr=thr,damping=damping)

        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)  



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
################################ Noise Contrastive Estimation ################################

# def MAP(sol,y,solpool,minimize=True):
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
# def MAP_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
#     sol = batch_solve(spsolver, y_hat,relaxation =False)
#     y = y_hat 
#     return MAP(sol,y,solpool,minimize)
# def MAP_hatc_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
#     sol = batch_solve(spsolver, y_hat,relaxation =False)
#     y = y_hat - y_true
#     return MAP(sol,y,solpool,minimize)

# def NCE(sol,y,solpool,minimize=True):
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
# def NCE_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
#     sol = batch_solve( spsolver, y_hat,relaxation =False)
#     y = y_hat 
#     return NCE(sol,y,solpool,minimize)
# def NCE_hatc_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
#     sol = batch_solve(spsolver, y_hat,relaxation =False)
#     y = y_hat - y_true
#     return NCE(sol,y,solpool,minimize)



# def growpool_fn(solver, solpool, y_hat):
#     '''
#     solpool is torch array [currentpoolsize,48]
#     y_hat is  torch array [batch_size,48]
#     '''
#     sol = batch_solve(solver, y_hat,relaxation =False).detach().numpy()
#     solpool_np = solpool.detach().numpy()
#     solpool_np = np.unique(np.append(solpool_np,sol,axis=0),axis=0)
#     # torch has no unique function, so we have to do this
#     return torch.from_numpy(solpool_np).float()
