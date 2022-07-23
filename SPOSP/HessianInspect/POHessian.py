import os
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

def eigen_plot(hess,figname='example'):
    import matplotlib.pyplot as plt
    from numpy import linalg as LA
    
    w, v = LA.eig(hess)
    w = np.sort(w, axis=None) 
    # plt.hist( w,bins= int(max(w))+1)
    plt.bar(w,np.ones_like(w),width=0.05)
    plt.ylabel('', fontsize=14, labelpad=10)
    plt.xlabel('Eigevalues', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    plt.savefig('{}.png'.format(figname))

def hessian(model,criterion, x_data,y_data,figname='example'):
    '''
    x_data,y_data are given as numpy arrays
    return hessian as numpy matrix
    '''
    layer = 0
    for param in model.net.parameters():
        y_pred = model(torch.from_numpy(x_data).float()).squeeze()
        env_loss = criterion( y_pred, torch.from_numpy(y_data).float())
        env_grads = torch.autograd.grad(env_loss, param, retain_graph=True, create_graph=True, allow_unused=True)
        len_grad = len(env_grads[0].flatten())
        flatten_grads = env_grads[0].flatten()
        print("grad shape",env_grads[0].shape)
        # row_size = env_grads[0].size(0)
        # col_size = env_grads[0].size(1)
        hess_params = torch.zeros(len_grad, len_grad)

        # for i in range(row_size):
        #     for j in range(col_size):
        #         hess_params[i*col_size + j] = torch.autograd.grad(env_grads[0][i][j], param, retain_graph=True,  allow_unused=True)[0].flatten() #
        #         hess_params[i*col_size + j] = torch.autograd.grad(env_grads[0][i][j], param, retain_graph=True,  allow_unused=True)[0].flatten() #

        for i in range(len_grad):
                hess_params[i] = torch.autograd.grad(flatten_grads[i], param, retain_graph=True,  allow_unused=True)[0].flatten() #



        eigen_plot(hess_params.detach().numpy(),figname=figname+'_layer'+str(layer))
        layer +=1
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
            return np.array(model.Xn).astype(np.float32)
        else:
            model = self.model
            sols = []
            for solindex in range(model.SolCount):
                model.setParam('SolutionNumber', solindex)
                sols.append(model.Xn)  
            sols = np.array(sols).astype(np.float32)
            print(sols.dot(y_true))
            return sols[np.argmax(sols.dot(y_true)*mm, axis=0)] 


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
            return torch.from_numpy(self.highest_regretsolution( y_hat.detach().numpy(),
                     y_true.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_hat)):
                solutions.append(torch.from_numpy(self.highest_regretsolution( y_hat[ii].detach().numpy(),
                     y_true[ii].detach().numpy())).float())
            return torch.stack(solutions)       
        
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
    def __init__(self, train_df,valid_df,test_df,generator, batch_size: int = 32, num_workers: int=8):
        super().__init__()
        self.train_df = train_df
        self.valid_df =  valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.generator =  generator
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batch_size, num_workers=self.num_workers)

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
        # regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii], sol_true[ii]) )
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii])[0] )
    return torch.mean( torch.tensor(regret_list ))

def regret_aslist(solver, y_hat,y_true, sol_true, minimize= True):  
    '''
    computes regret of more than one cost vectors
    ''' 
    regret_list = []
    for ii in range(len(y_true)):
        # regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii], sol_true[ii]).item() )
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y_true[ii]).item() )
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

def SPOLoss(solver= spsolver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true):
       
            sol_hat = solver.solution_fromtorch(y_pred)
            sol_spo = solver.solution_fromtorch(2* y_pred - y_true)
            sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            # return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret
            return mm*torch.mul((  sol_hat - sol_true) , y_true).sum()/ torch.mul(sol_true,y_true).sum(), sol_spo,  sol_true

        @staticmethod
        def backward(ctx, grad_output,grad_solspo,grad_soltrue):
            # print("grad_op",grad_output)
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return mm*(sol_true - sol_spo)*grad_output, None

    return SPOLoss_cls.apply

# def WorstcaseSPOLoss(solver, minimize=True):
#     mm = 1 if minimize else -1
#     class SPOLoss_cls(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, y_pred, y_true, sol_true):
       
#             # sol_hat = solver.solution_fromtorch(y_pred)
#             sol_hat = solver.highest_regretsolution_fromtorch(y_pred,y_true,minimize=True)
#             sol_spo = solver.solution_fromtorch(2* y_pred - y_true)
#             # sol_true = solver.solution_fromtorch(y_true)
#             ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
#             return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

#         @staticmethod
#         def backward(ctx, grad_output):
#             sol_spo,  sol_true, sol_hat = ctx.saved_tensors
#             return mm*(sol_true - sol_spo), None, None
            
#     return SPOLoss_cls.apply



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
            # loss += self.loss_fn(self.exact_solver)(y_hat[ii],y[ii], sol[ii])
            loss += self.loss_fn(self.exact_solver)(y_hat[ii],y[ii])[0]
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


