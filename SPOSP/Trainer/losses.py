import logging
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np


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

