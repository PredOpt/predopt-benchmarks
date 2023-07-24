import logging
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from Trainer.utils import batch_solve

def SPOlayer(solver,minimize = False):
    mm = 1 if minimize else -1
    class SPOlayer_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat,y_true,sol_true ):
            sol_hat = batch_solve(solver, y_hat)

            ctx.save_for_backward(y_hat,y_true,sol_true)

            return ( mm*(sol_hat -sol_true)*y_true).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat,y_true,sol_true = ctx.saved_tensors
            y_spo = 2*y_hat - y_true
            sol_spo = batch_solve(solver,y_spo) 
            return (sol_true - sol_spo)*mm, None, None
    return SPOlayer_cls.apply


def DBBlayer(solver,lambda_val=1., minimize = False):
    mm = 1 if minimize else -1
    class DBBlayer_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat,y_true,sol_true ):
            sol_hat =  batch_solve(solver, y_hat) 

            ctx.save_for_backward(y_hat,y_true,sol_true, sol_hat)

            return sol_hat

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we compute gradient to minimize regret
            """
            y_hat,y_true,sol_true, sol_hat= ctx.saved_tensors
            y_perturbed = y_hat + mm* lambda_val* grad_output
            sol_perturbed =  batch_solve(solver, y_perturbed) 
            
            return -mm*(sol_hat - sol_perturbed)/lambda_val, None, None
    return DBBlayer_cls.apply