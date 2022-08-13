import logging
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from Trainer.optmizer_module import G
import networkx as nx
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
from intopt.intopt_model import IPOfunc
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model

def batch_solve(solver, y,relaxation =False):
    sol = []
    for i in range(len(y)):
        sol.append(   solver.solution_fromtorch(y[i]).reshape(1,-1)   )
    return torch.cat(sol,0)


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


###################################### Ranking Loss  Functions  #########################################

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



def pointwise_loss(y_hat,y_true,sol_true, cache,*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    f(y_hat,s) is regresson on f(y,s)
    '''
    loss = (torch.matmul(y_hat,cache.transpose(0,1))- torch.matmul(y_true,cache.transpose(0,1))).square().mean()

    return loss



def pairwise_loss(y_hat,y_true,sol_true, cache,tau=0, minimize=True,mode= 'B'):
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

def pairwise_diffloss(y_hat,y_true,sol_true, cache,tau=0, minimize=True,mode= 'B'):
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

def Listnet_loss(y_hat,y_true,sol_true, cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += -(F.log_softmax((-mm*y_hat[ii]*cache/tau).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss
def Listnet_KLloss(y_hat,y_true,sol_true,cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += ( F.log_softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0) -
         F.log_softmax((-mm*y_hat[ii]*cache).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss

def MAP(y_tilde,sol_true, cache,minimize=True):
    '''
    sol, and y are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y_tilde)):
        loss += torch.max(((sol_true[ii] - cache )*(mm*y_tilde[ii]  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,sol_true, cache,minimize=True,*wd,**kwd):
    y_tilde = y_hat 
    return MAP( y_tilde, sol_true, cache,minimize)
def MAP_hatc_c(y_hat,y_true,sol_true, cache,minimize=True,*wd,**kwd):
    y_tilde= y_hat - y_true
    return MAP(y_tilde, sol_true, cache,minimize)

def NCE(y_tilde,sol_true, cache,minimize=True):
    '''
    sol, and y are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y_tilde)):
        loss += torch.mean(((sol_true[ii] - cache )*(mm*y_tilde[ii]  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,sol_true,cache,minimize=True,*wd,**kwd):
    y_tilde = y_hat 
    return NCE(y_tilde, sol_true, cache,minimize)
def NCE_hatc_c(y_hat,y_true,sol_true,cache,minimize=True,*wd,**kwd):
    y_tilde = y_hat - y_true
    return NCE(y_tilde, sol_true, cache,minimize)

