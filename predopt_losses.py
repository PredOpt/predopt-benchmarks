import torch
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers


def SPOLoss(solver, sign=1):
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):
            sol_hat = solver.solve_from_torch(input)
            sol_spo = solver.solve_from_torch(2*input - target)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return sign*(sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return sign*(sol_true - sol_spo), None, None

    return SPOLoss_cls.apply

def SPOLoss_tiebreak(solver, sign=1):
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):
            sol_hat = solver.solve_from_torch(input)[0]
            sols_spo = solver.solve_from_torch(2*input - target)
            id_worse_sol_spo = (sols_spo @ target).argmin()
            sol_spo = sols_spo[id_worse_sol_spo]
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return sign*(sol_hat[0] - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return sign*(sol_true - sol_spo), None, None

    return SPOLoss_cls.apply


def BlackBoxLoss(solver, mu=0.1, sign=1):
    class BlackboxLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):
            sol_hat = solver.solve_from_torch(input)
            sol_perturbed = solver.solve_from_torch(input + mu * target)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return sign*(sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -sign*(sol_hat - sol_perturbed)/mu, None, None

    return BlackboxLoss_cls.apply


def NCECacheLoss(variant: int, sign=1):
    def get_loss(pred, target, sol_true, cache_sols):
        pred = pred.view(*target.shape)
        if variant == 1:
            loss = ((sign*(cache_sols - sol_true)*pred).sum())
        if variant == 2:
            loss = ((sign*(cache_sols - sol_true)
                    * (pred - target)).sum())
        if variant == 3:
            loss = (sign*(cache_sols - sol_true)
                    * pred).sum(dim=1).max()
        if variant == 4:
            loss = (sign*(cache_sols - sol_true)*(pred -
                                                  target)).sum(dim=1).max()
        return loss
    return get_loss                                                                                                                                                                                           


def QPTLoss(A_trch, b_trch, G_trch, h_trch, Q_trch, model_params_quad, sign=1):
    def get_loss(input, target, sol_true):
        sol_hat_qp = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
                                model_params=model_params_quad)(Q_trch.expand(1, *Q_trch.shape),
                                                                sign *
                                                                input, G_trch.expand(
                                                                    1, *G_trch.shape),
                                                                h_trch.expand(
                                                                    1, *h_trch.shape),
                                                                A_trch.expand(1, *A_trch.shape), b_trch.expand(1, *b_trch.shape)).squeeze()
        return (sol_hat_qp - sol_true).dot(target)

    return get_loss
