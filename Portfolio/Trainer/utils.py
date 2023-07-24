import torch 
import numpy as np

def batch_solve(solver, y,relaxation =False):
    sol = []
    for i in range(len(y)):
        sol.append(   solver.solution_fromtorch(y[i]).reshape(1,-1)   )
    return torch.cat(sol,0).float()


def regret_list(solver, y_hat,y_true, sol_true, minimize = False):  
    '''
    computes regret of more than one cost vectors
    ''' 
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver, y_hat )
    return ((mm*(sol_hat - sol_true)*y_true).sum(1)/ (sol_true*y_true).sum(1) )
def abs_regret_list(solver,y_hat,y_true,sol_true,minimize = False):
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver, y_hat )
    return ((mm*(sol_hat - sol_true)*y_true).sum(1) )

def regret_fn(solver, y_hat,y_true, sol_true, minimize = False):  
    ### Converting infinity to 1, there are lots of innities where all the returns are negative
    return torch.nan_to_num( regret_list(solver, y_hat,y_true, sol_true, minimize= minimize),  nan=0., posinf=1.).mean()

def abs_regret_fn(solver, y_hat,y_true, sol_true, minimize = False):  
    return abs_regret_list(solver, y_hat,y_true, sol_true, minimize= minimize).mean()


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