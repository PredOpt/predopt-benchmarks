import numpy as np
import torch
import torch.nn as nn
from Trainer.utils import batch_solve



def SPOlayer(solver,minimize=False):
    mm = 1 if minimize else -1
    class SPOlayer_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat,y_true,sol_true ,m ):
            sol_hat = batch_solve(solver, y_hat, m)

            ctx.save_for_backward(y_hat,y_true,sol_true ,m )

            return ( mm*(sol_hat -sol_true)*y_true).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat,y_true,sol_true ,m = ctx.saved_tensors
            y_spo = 2*y_hat - y_true
            sol_spo = batch_solve(solver,y_spo ,m)
            return (sol_true - sol_spo)*mm, None, None, None
    return SPOlayer_cls.apply


def DBBlayer(solver,lambda_val=1., minimize=False):
    mm = 1 if minimize else -1
    class DBBlayer_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat,y_true,sol_true ,m ):
            sol_hat = batch_solve(solver, y_hat ,m)

            ctx.save_for_backward(y_hat,y_true,sol_true, sol_hat ,m)

            return sol_hat

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we compute gradient to minimize regret
            """
            y_hat,y_true,sol_true, sol_hat ,m = ctx.saved_tensors
            y_perturbed = y_hat + mm* lambda_val* grad_output
            sol_perturbed = batch_solve(solver, y_perturbed ,m)
            
            return -mm*(sol_hat - sol_perturbed)/lambda_val, None, None, None
    return DBBlayer_cls.apply