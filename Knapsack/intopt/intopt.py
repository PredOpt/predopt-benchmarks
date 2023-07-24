import numpy as np
import scipy as sp
import scipy.sparse as sps
import numbers
from warnings import warn
from scipy.linalg import LinAlgError
import torch
from torch import nn, optim
from torch.autograd import Variable,Function
import sys
import logging
import time, datetime
import warnings
from scipy.optimize import OptimizeWarning
#from intopt.remove_redundancy import _remove_redundancy, _remove_redundancy_sparse, _remove_redundancy_dense
np.set_printoptions(threshold=np.inf)
from intopt.presolve import presolve

from intopt.util import convert_to_np, standardizeLP
from intopt.solveLP import solveLP
############################ Code Adapted from https://github.com/scipy/scipy/blob/5dcc0f66fe6af9d954d1a7e3c0f451736fa7500a/scipy/optimize/_linprog_ip.py ####


"""
References
----------
     Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""
def intopt_nonbacthed(A_trch =None,b_trch =None,G_trch =None,h_trch =None, thr= 1e-8, damping= 1e-5, diffKKT = False, dopresolve=True):
    '''

    A : 2D tensor, optional
        The equality constraint matrix. 
    b : 1D tensor, optional
        The equality constraint vector. 
    G : 2D tensor, optional
        The inequality constraint matrix. 
    h : 1D tensor, optional
        The inequality constraint vector.
    thr: lambda_cutoff parameter, optional
        default value 1e-5
    damping: Tikhonov damping parameter, optional
        default value 1e-5   
    diffKKT: boolean,
            if True, differentiate the KKT conditions, else the HSD embedding
    dopresolve: boolean
                if not True no presolving before solving the LP

    The feasible region is defined by
            G @ x <= h
            A @ x == b 
    Bounds other than x >=0 must be specified in the inequality matrix G @ x <= h
    No need to specify x>=0 constraints
    '''
    run_time = 0.
    
    # if no solution under timelimit or max-iter don't do gradient update
    A,b,G, h = convert_to_np(A_trch, b_trch, G_trch, h_trch)
    if dopresolve:

        presolver = presolve (G,h, A,b)
        (G,h, A,b) = presolver.transform ()

   
    standardizer = standardizeLP (G,h, A,b)
    A_,b_ = standardizer.getAb()
    

    class WrappedFunc_cls(Function):        
        @staticmethod
        def forward(ctx,c_trch):
            '''
            In the forward pass take the objective function parameter c
            c : 1D array (n_x,)
                The coefficients of the linear objective function to be minimized.
            '''
            nonlocal run_time
            start = time.time()
            c = c_trch.detach().numpy()

            c_ = standardizer.transformC(c) 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y, z, tau, kappa, mu = solveLP(c_ ,A_,b_, thr )
            
            x_solve = torch.from_numpy( standardizer.transformsolution (x/tau)).float()

            ctx.c = c_
            ctx.A = A_
            ctx.b = b_
            ctx.x = x
            ctx.y = y
            ctx.z = z
            ctx.tau = tau
            ctx.kappa = kappa


            end = time.time()
            run_time += end -start
            return x_solve


        @staticmethod
        def backward(ctx,dx):
            '''
            The backward pass differentiate the HSD embedding to compute dx/dc
            and then multiply with dx to return del_c
            Eq. 12 in the paper is solved in this function.
            We use X^{-1} Z = mu X^{-2}.
            M = A Z^{-1} X A.T = A X^{2} A.T \ mu

                    |    -X^{-1}Z       |   A.T         |
                W=  |-----------------------------------|
                    |       A           |   0           |


            '''
            nonlocal run_time
            start = time.time()
            ### we first compute the drivative of the augmented x_vector 

            c_ = ctx.c
            A_  = ctx.A
            b_  = ctx.b
            x = ctx.x
            y = ctx.y
            z = ctx.z
            tau = ctx.tau
            kappa = ctx.kappa

            n  = len(x)
            


            #### Tikhonov damped
            
            if diffKKT:
                """
                W [dx dy].T =  [I 0].T
                M  dy = A X^{2}  /mu
                dx =  X^{2} (A.T dy - I)
                """
                x =  x/tau
                y = y/tau
                z = z/tau

                mu = (x.dot(z)) / n
                Dinv = x / z
                
                M = A_ .dot(Dinv.reshape(-1, 1) * A_ .T) #### we don't divide by mu because it would cancel out  
                np.fill_diagonal(M, M.diagonal() + damping)  


                r = A_ *Dinv
                dely = sp.linalg.solve( M ,r ,assume_a='pos')
                delx = (np.matmul((Dinv.reshape(-1, 1)*A_ .T ), dely ) -  np.diag(Dinv))
                delx_torch = torch.from_numpy( standardizer.transformsgradient (delx)).float()
            
            else:
                """
                W [w1 t1].T = [c b].T
                W [w2 t2].T = [tau*I 0].T
                [w1, t1, w2, t2] found out by:
                M t1 = b + A X^{2} c/ mu; w1 = X^{2} (A.T t1 - c)/mu
                M t2 = A X^{2} tau /mu; w2 = X^{2} (A.T t2 - tau*I)/mu

                dtau = (x + w2.T @ c - t2.T @ b)/ (kappa/tau + b.dot(t1) - c.dot(w1))
                dx = w2 + w1 @ dtau
                """
                mu = (x.dot(z) + tau*kappa) / (n + 1)
                Dinv = x / z
                
                M = A_.dot(Dinv.reshape(-1, 1) * A_ .T) #### we don't divide by mu because it would cancel out    
                np.fill_diagonal(M, M.diagonal() + damping) 
                      
                ##########   Solve :  M t1 = b +  A X^2 c /mu
                r1 =  b_  +  A_ .dot(Dinv*c_ )
                t1 = sp.linalg.solve( M ,r1 ,assume_a='pos')
                #### w1  = X^2 (A.T @ t1 - c)\mu 
                w1  = (A_.T.dot(t1) - c_ )*Dinv
                ##########   Solve :  M t2 =   A X^2 tau /mu
                r2 = tau*A_ *Dinv
                t2 = sp.linalg.solve( M ,r2 ,assume_a='pos')
                #### w2  = (X^2 @ A.T @ t2 - tau X^2 ) /mu 
                w2 = (np.matmul((Dinv.reshape(-1, 1)*A_ .T ),t2) - tau* np.diag(Dinv))
                ###### deltau = x + w2.T @ c - t2 .T @ b / (-c.dot(w1)+ b.dot(t1) + kappa/tau)

                deltau = (x + w2.T.dot(c_ ) - t2.T.dot(b_ ))/( b_ .dot(t1) - c_ .dot(w1)  + (kappa/tau)  )
                ####  delx = w2 + w1.T @ deltau
                delx = w2 +  np.einsum('i,j->ij', w1 ,deltau)

                ### This is the differentiation of the augmented x in HSD
                ### Actual solution x(actual) = x(hsd)/ tau
                ### dx(actual) = (tau* dx- x* dtau)/tau^2

                delx = (tau*delx - x*deltau)/(tau**2)
                delx_torch = torch.from_numpy( standardizer.transformsgradient (delx)).float()
            
            return torch.matmul(delx_torch, dx)            
    return WrappedFunc_cls.apply
    

class intopt(nn.Module):
    '''
        Batched Implementation of the Above Module
        For now, I am just iterating using a for loop
    '''
    def __init__(self,A_trch= None,b_trch= None,G_trch= None,h_trch= None, thr= 1e-8, damping= 1e-5, diffKKT = False, dopresolve = True):
        super().__init__()
        self.A_trch, self.b_trch,  self.G_trch, self.h_trch = A_trch ,b_trch ,G_trch ,h_trch 
        self.thr, self.damping = thr, damping
        self.net =  intopt_nonbacthed(A_trch ,b_trch ,G_trch ,h_trch , thr, damping, diffKKT, dopresolve )
    def forward(self, c_trch):
        '''
        In the forward pass take the objective function parameter c
        c : 2D array (batch_size, n_x)
        '''
        batch_size = len (c_trch)
        sol = torch.zeros_like(c_trch)
        for i in range(batch_size):
            sol[i] = self.net(c_trch[i])
        return sol
    