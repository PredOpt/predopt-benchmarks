import numpy as np
import torch
import torch.nn as nn
from Trainer.utils import batch_solve
def SPOlayer(solver,minimize=False):
    mm = 1 if minimize else -1
    class SPOlayer_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat,y_true,sol_true ):

            ctx.save_for_backward(y_hat,y_true,sol_true)

            return ( mm*(y_hat -y_true)*sol_true).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat,y_true,sol_true = ctx.saved_tensors
            y_spo = 2*y_hat - y_true
            sol_spo = batch_solve(solver,y_spo)
            return (sol_true - sol_spo)*mm, None, None
    return SPOlayer_cls.apply