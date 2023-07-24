import torch
import numpy as np



def batch_solve(solver,y):
    '''
    wrapper around te solver to return solution of a vector of cost coefficients
    '''
    sol = []
    for i in range(len(y)):
        sol.append( solver.solve(y[i]))
    return torch.from_numpy( np.array(sol) ).float()


def regret_aslist(solver, y_hat,y_true, sol_true, minimize=True): 
    '''
    computes regret of more than one cost vectors
    ''' 
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver,y_hat.detach().numpy())
    return  ((mm*(sol_hat - sol_true)*y_true).sum(1)/(sol_true*y_true).sum(1))

def regret_fn(solver, y_hat,y_true, sol_true, minimize=True):
    '''
    computes average regret given a predicted cost vector and the true solution vector and the true cost vector
    y_hat,y, sol_true are torch tensors
    '''
    return regret_aslist(solver,y_hat,y_true,sol_true,minimize).mean()


def abs_regret_aslist(solver, y_hat,y_true, sol_true, minimize=True): 
    '''
    computes regret of more than one cost vectors
    ''' 
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver,y_hat.detach().numpy())
    return  ((mm*(sol_hat - sol_true)*y_true).sum(1))


def abs_regret_fn(solver, y_hat,y_true, sol_true, minimize=True):
    '''
    computes average regret given a predicted cost vector and the true solution vector and the true cost vector
    y_hat,y, sol_true are torch tensors
    '''
    return abs_regret_aslist(solver,y_hat,y_true,sol_true,minimize).mean()


def growpool_fn(solver,cache, y_hat):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol = batch_solve(solver,y_hat)
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,sol,axis=0),axis=0)
    # torch has no unique function, so we have to do this
    return torch.from_numpy(cache_np).float()