from abc import abstractmethod
# from qpthlocal.qp import make_gurobi_model
# from qpthlocal.qp import QPSolvers
# from qpthlocal.qp import QPFunction
# from cvxpylayers.torch import CvxpyLayer
# import cvxpylayers
import cvxpy as cp
import torch
from torch import nn, optim
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
import pytorch_lightning as pl

from predopt_losses import BlackBoxLoss, SPOLoss
# import numpy as np


class Solver:
    """Abstract Class

    Wrapper class for specifying and solving a problem

    """
    @abstractmethod
    def solve_from_torch(self, y_torch:torch.Tensor):
        """Solve the problem for a given cost vector

        Args: 
            `y_torch`: cost vector as a PyTorch tensor

        Returns:
            vector of decision variables as a PyTorch Float tensor
        """
        pass
    
    @abstractmethod
    def get_constraints_matrix_form(self):
        """Return linear constraints in matrix form, such that for decision variables `x`:

            `Ax = b` and
            `Gx <= h` 
        
        Returns:
            (`A`,`b`,`G`,`h`) tuple. If the problem does not have (in)equality constraints, related tensors are set to `None`
        """
        pass


class Datawrapper:
    def __init__(self, x,y, solver:Solver):
        self.x = x
        self.y = y
        self.sol = torch.stack([solver.solve_from_torch(yi) for yi in y])

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sol[index]






class TwoStageRegression(pl.LightningModule):
    def __init__(self, net:nn.Module, solver:Solver, lr=1e-1):
        super().__init__()
        self.net = net
        self.lr = lr
        self.solver = solver
        self.save_hyperparameters("lr")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x, y, sol_true = batch
        y_hat = self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_list = []
        calc_regret = SPOLoss(self.solver)
        for ii in range(len(y)):
            regret_list.append(calc_regret(y_hat[ii], y[ii], sol_true[ii]))
        regret_loss = torch.mean(torch.tensor(regret_list))

        self.log("val_mse", mseloss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        return mseloss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SPO(TwoStageRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.solver = solver
        self.po_criterion = SPOLoss(self.solver)

    def training_step(self, batch, batch_idx):
        x, y, sol_true = batch
        y_hat = self(x).squeeze()
        loss = 0
        for ii in range(len(y)):
            loss += self.po_criterion(y_hat[ii], y[ii], sol_true)
        return loss/len(y)


class Blackbox(SPO):
    def __init__(self,*args, mu=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.po_criterion = BlackBoxLoss(self.solver, mu)


