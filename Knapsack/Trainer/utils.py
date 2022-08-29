import torch
import numpy as np


def batch_solve(solver, y):

    sol = []
    for i in range(len(y)):
        sol.append(  solver.solve(y[i].detach().numpy()) )
    return torch.tensor(sol).float()

def regret_list(solver, y_hat,y_true, sol_true, minimize=False):
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver,y_hat)
    
    return (mm*(sol_hat - sol_true)*y_true).sum(1)


def regret_fn(solver, y_hat,y_true, sol_true, minimize=False):

    
    return regret_list(solver,y_hat,y_true,sol_true,minimize).mean()