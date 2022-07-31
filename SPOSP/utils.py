import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from losses import SPOLoss, batch_solve



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