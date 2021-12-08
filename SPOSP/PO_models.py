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
        y the vector of  edge weight
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
            print("Objective", model.objVal)
            return x.x
    def solution_fromtorch(self,y_torch):
        return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()

###################################### Wrapper #########################################
class datawrapper():
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def SPOLoss(solver):
    solver =  shortestpath_solver()
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target):
       
            sol_hat = solver.solution_fromtorch(input)
            sol_spo = solver.solution_fromtorch(2*input - target)
            sol_true = solver.solution_fromtorch(target)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            

            return   (  sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return sol_true - sol_spo, None
            
    return SPOLoss_cls.apply



class twostage_regression(pl.LightningModule):
    def __init__(self,net,lr=1e-1):
        super().__init__()
        self.net =  net
        self.lr = lr
        self.save_hyperparameters("lr")
    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y = batch
        y_hat =  self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_list = []
        for ii in range(len(y)):
            regret_list.append( SPOLoss()(y_hat[ii],y[ii]) )
        regret_loss = torch.mean( torch.tensor(regret_list ))

        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return mseloss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




class SPO(twostage_regression):
    def __init__(self,net,lr=1e-1):
        super().__init__(net,lr)
        # self.automatic_optimization = True
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0

        for ii in range(len(y)):
            loss += SPOLoss()(y_hat[ii],y[ii])
        return loss