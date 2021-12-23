from abc import abstractmethod
# from qpthlocal.qp import make_gurobi_model
# from qpthlocal.qp import QPSolvers
# from qpthlocal.qp import QPFunction
# from cvxpylayers.torch import CvxpyLayer
# import cvxpylayers
import cvxpy as cp
from numpy.core.numeric import indices
import torch
from torch import Tensor, nn, optim
import numpy as np
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
import pytorch_lightning as pl

from predopt_losses import BlackBoxLoss, SPOLoss, NCECacheLoss, QPTLoss
# import numpy as np
from qpthlocal.qp import make_gurobi_model



class Solver:
    """Abstract Class

    Wrapper class for specifying and solving a problem

    """
    
    def solve_from_torch(self, y_torch:torch.Tensor):
        """Solve the problem for a given cost vector

        Args: 
            `y_torch`: cost vector as a PyTorch tensor

        Returns:
            vector of decision variables as a PyTorch Float tensor
        """
        return torch.from_numpy(self.solve(y_torch.detach().numpy())).float()
    
    @abstractmethod
    def solve(self, y: np.ndarray):
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
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        self.y = y if isinstance(x, torch.Tensor) else torch.from_numpy(y)
        self.sol = torch.stack([solver.solve_from_torch(yi) for yi in self.y])

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sol[index]



class TwoStageRegression(pl.LightningModule):
    def __init__(self, net:nn.Module, solver:Solver, lr=1e-1, twostage_criterion=nn.MSELoss(reduction='mean'), maximize=False):
        super().__init__()
        self.net = net
        self.lr = lr
        self.solver = solver
        self.criterion = twostage_criterion
        self.maximize = maximize
        self.save_hyperparameters("lr", "twostage_criterion")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        # criterion = nn.MSELoss(reduction='mean')
        x, y, sol_true = batch
        y_hat = self(x).squeeze()
        mseloss = self.criterion(y_hat.view(y.shape), y)
        regret_list = []
        calc_regret = SPOLoss(self.solver)
        for ii in range(len(y)):
            regret_list.append(calc_regret(y_hat.view(y.shape)[ii], y[ii], sol_true[ii]))
        regret_loss = torch.mean(torch.tensor(regret_list))

        self.log("val_mse", mseloss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True,
                 on_step=True, on_epoch=True, )
        return {
            'val_mse': mseloss,
            'val_regret':regret_loss
        }
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_mse"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_regret"] for x in outputs]).mean()
        self.log("ptl/val_mse", avg_loss)
        self.log("ptl/val_regret", avg_acc)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SPO(TwoStageRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.po_criterion = SPOLoss(self.solver)

    def training_step(self, batch, batch_idx):
        x, y, sol_true = batch
        y_hat = self(x).squeeze()
        loss = 0
        for ii in range(len(y)):
            loss += self.po_criterion(y_hat[ii], y[ii], sol_true[ii])
        return loss/len(y)


class Blackbox(SPO):
    def __init__(self,*args, mu=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.po_criterion = BlackBoxLoss(self.solver, mu)
        self.save_hyperparameters('mu')


class NCECache(SPO):
    def __init__(self, *args, cache_sols=None, psolve=0.0, variant=1, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psolve = psolve
        self.cache_sols = cache_sols
        self.po_criterion = NCECacheLoss(variant)
        self.save_hyperparameters('psolve', 'variant')
        self.rng = np.random.default_rng(seed)

    def training_step(self, batch, batch_idx):
        x, y, sol_true = batch 
        y_hat = self(x).squeeze()
        loss = 0
        for i in range(len(y)):
            y_hat_i = y_hat[i]
            if self.rng.binomial(1, self.psolve) > 0:
                sol_hat_i = self.solver.solve_from_torch(y_hat_i)
                self.cache_sols = torch.cat((sol_hat_i.unsqueeze(0), self.cache_sols)).unique(dim=0)
            loss += self.po_criterion(y_hat_i, y[i], sol_true[i], self.cache_sols)
        return loss / len(y)
            

class QPTL(SPO):
    def __init__(self, *args, tau=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau 
        self.save_hyperparameters('tau')
        A, b, G, h = self.solver.get_constraints_matrix_form()
        G_trch = torch.from_numpy(G if G is not None else np.random.randn(1,0)).float()
        h_trch = torch.from_numpy(h if h is not None else np.array([])).float()
        A_trch = torch.from_numpy(A if A is not None else np.random.randn(1,0)).float()
        b_trch = torch.from_numpy(b if b is not None else np.array([])).float()
        Q_trch = (self.tau)*torch.eye(G.shape[1] if G is not None else A.shape[1])
        model_params_quad = make_gurobi_model(G, h, 
            A, b, Q_trch.detach().numpy() )

        self.po_criterion = QPTLoss(A_trch, b_trch, G_trch, h_trch, Q_trch, model_params_quad)
if __name__ == '__main__':
    from SPOSP.train import train_dl, test_dl
    from SPOSP.solver import spsolver
    trainer = pl.Trainer(max_epochs= 1,  min_epochs=4)
    # model = TwoStageRegression(net=nn.Linear(5,1), solver=spsolver, lr= 0.01)
    # model = NCECache(cache_sols=train_df.sol, net=nn.Linear(5,1), solver=spsolver, lr=0.001, psolve=0.1, seed=243)
    model = QPTL(net=nn.Linear(5,1), solver=spsolver, lr= 0.01, tau=10)
    trainer.fit(model, train_dl,test_dl)
    result = trainer.test(test_dataloaders=test_dl)