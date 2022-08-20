from Trainer.bipartite import  bmatching_diverse, get_qpt_matrices
import torch
import numpy as np

# solver = bmatching_diverse
# objective_fun=lambda x,v,**params: x @ v

def batch_solve(solver,y,m,relaxation =False,batched= True):

    if batched:
        ### y, m both are of dim (*,2500)
        sol = []

        for i in range(len(y)):
            sol.append(  solver.solve(y[i].detach().numpy(), m[i].numpy(), relaxation=relaxation) )
        return torch.tensor(sol).float()
    else:
        ### y, m both are of dim (2500)
        sol = solver.solve(y.detach().numpy(), m.numpy(), relaxation=relaxation)
        return torch.tensor(sol).float()


def regret_list(solver,y_hat,y_true,sol_true,m,minimize=False):
    mm = 1 if minimize else -1    
    sol_hat = batch_solve(solver, y_hat,m)
    return ((mm*(sol_hat - sol_true)*y_true).sum(1)/ (sol_true*y_true).sum(1) )

def regret_fn(solver,y_hat,y_true,sol_true,m,minimize=False):
    # mm = 1 if minimize else -1    
    # sol_hat = batch_solve(y_hat,m)
    # sol_ = batch_solve(y,m)
    # # return ((mm*(sol_hat - sol_)*y).sum(1)/ (sol_*y).sum(1) ).mean()
    return  regret_list(solver,y_hat,y_true,sol_true,m,minimize=minimize).mean()

# def growpool_fn(solpool, y_hat, m):
#     '''
#     solpool is torch array [currentpoolsize,48]
#     y_hat is  torch array [batch_size,48]
#     '''
#     sol = batch_solve(y_hat,m).detach().numpy()
#     solpool_np = solpool.detach().numpy()
#     solpool_np = np.unique(np.append(solpool_np,sol,axis=0),axis=0)
#     # torch has no unique function, so we have to do this
#     return torch.from_numpy(solpool_np).float()