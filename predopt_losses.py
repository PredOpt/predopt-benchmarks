import torch


def SPOLoss(solver):
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):

            sol_hat = solver.solve_from_torch(input)
            sol_spo = solver.solve_from_torch(2*input - target)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return (sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return sol_true - sol_spo, None

    return SPOLoss_cls.apply


def BlackBoxLoss(solver, mu=0.1):
    class BlackboxLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):

            sol_hat = solver.solve_from_torch(input)
            sol_perturbed = solver.solve_from_torch(input + mu * target)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return (sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -(sol_hat - sol_perturbed)/mu, None

    return BlackboxLoss_cls.apply


def NCECacheLoss(variant:int):
    def forward(pred, target, sol_true, cache_sols):
        pred = pred.view(*target.shape)
        if variant == 1:  
            loss = (((cache_sols - sol_true)*pred).sum())
        if variant == 3: 
            loss = (((cache_sols - sol_true)
                    * (pred - target)).sum())
        if variant == 4:  
            loss = ((cache_sols - sol_true)
                    * pred).sum(dim=1).max()
        if variant == 5:  
            loss = ((cache_sols - sol_true)*(pred -
                                                            target)).sum(dim=1).max()
        return loss
    return forward
