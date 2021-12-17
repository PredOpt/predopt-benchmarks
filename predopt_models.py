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



def SPOLoss(solver:Solver):
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target, sol_true):

            sol_hat = solver.solve_from_torch(input)
            sol_spo = solver.solve_from_torch(2*input - target)
            #sol_true = solver.solve_from_torch(target)
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
            #sol_true = solver.solve_from_torch(target)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return (sol_hat - sol_true).dot(target)

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -(sol_hat - sol_perturbed)/mu, None

    return BlackboxLoss_cls.apply


class TwoStageRegression(pl.LightningModule):
    def __init__(self, net:nn.Module, exact_solver:Solver, lr=1e-1):
        super().__init__()
        self.net = net
        self.lr = lr
        self.exact_solver = exact_solver
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
        calc_regret = SPOLoss(self.exact_solver)
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


# class SPO(twostage_regression):
#     def __init__(self, net, solver=spsolver, exact_solver=spsolver, lr=1e-1):
#         super().__init__(net, exact_solver, lr)
#         self.solver = solver

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x).squeeze()
#         loss = 0

#         for ii in range(len(y)):
#             loss += SPOLoss(self.solver)(y_hat[ii], y[ii])
#         return loss/len(y)


# class Blackbox(twostage_regression):
#     def __init__(self, net, solver=spsolver, exact_solver=spsolver, lr=1e-1, mu=0.1):
#         super().__init__(net, exact_solver, lr)
#         self.mu = mu
#         self.solver = solver

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x).squeeze()
#         loss = 0

#         for ii in range(len(y)):
#             loss += BlackboxLoss(self.solver, self.mu)(y_hat[ii], y[ii])
#         return loss/len(y)


###################################### This approach use it's own solver #########################################

# Build cvxpy modle prototype

# class cvxsolver:
#     def __init__(self, G=G):
#         self.G = G

#     def make_proto(self):
#         G = self.G
#         num_nodes, num_edges = G.number_of_nodes(),  G.number_of_edges()
#         A = cp.Parameter((num_nodes, num_edges))
#         b = cp.Parameter(num_nodes)
#         c = cp.Parameter(num_edges)
#         x = cp.Variable(num_edges)
#         constraints = [x >= 0, x <= 1, A @ x == b]
#         objective = cp.Minimize(c @ x)
#         problem = cp.Problem(objective, constraints)
#         self.layer = CvxpyLayer(problem, parameters=[A, b, c], variables=[x])

#     def shortest_pathsolution(self, y):
#         self.make_proto()
#         G = self.G
#         A = torch.from_numpy(
#             (nx.incidence_matrix(G, oriented=True).todense())).float()
#         b = torch.zeros(len(A))
#         b[0] = -1
#         b[-1] = 1
#         sol, = self.layer(A, b, y)
#         return sol

#     def solve_from_torch(self, y_torch):
#         return self.shortest_pathsolution(y_torch.float())


# solver = cvxsolver()


# class qpsolver:
#     def __init__(self, G=G, mu=1e-6):
#         self.G = G
#         A = nx.incidence_matrix(G, oriented=True).todense()
#         b = np.zeros(len(A))
#         b[0] = -1
#         b[-1] = 1
#         self.mu = mu
#         G_lb = -1*np.eye(A.shape[1])
#         h_lb = np.zeros(A.shape[1])
#         G_ub = 1*np.eye(A.shape[1])
#         h_ub = np.ones(A.shape[1])
#         G_ineq = np.concatenate((G_lb, G_ub))
#         h_ineq = np.concatenate((h_lb, h_ub))

#         self.model_params_quad = make_gurobi_model(G_ineq, h_ineq,
#                                                    A, b,  mu*np.eye(A.shape[1]))
#         self.solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
#                                  model_params=self.model_params_quad, eps=1e-8)

#     def shortest_pathsolution(self, y):
#         G = self.G
#         A = torch.from_numpy(
#             (nx.incidence_matrix(G, oriented=True).todense())).float()
#         b = torch.zeros(len(A))
#         b[0] = -1
#         b[-1] = 1
#         Q = self.mu*torch.eye(A.shape[1])
#         G_lb = -1*torch.eye(A.shape[1])
#         h_lb = torch.zeros(A.shape[1])
#         G_ub = torch.eye(A.shape[1])
#         h_ub = torch.ones(A.shape[1])
#         G_ineq = torch.cat((G_lb, G_ub))
#         h_ineq = torch.cat((h_lb, h_ub))

#         sol = self.solver(Q.expand(1, *Q.shape),
#                           y, G_ineq.expand(
#                               1, *G_ineq.shape), h_ineq.expand(1, *h_ineq.shape),
#                           A.expand(1, *A.shape), b.expand(1, *b.shape))
#         return sol

#     def solve_from_torch(self, y_torch):
#         return self.shortest_pathsolution(y_torch.float())


# solver = qpsolver()


# class DCOL(twostage_regression):
#     '''
#     Implementation of
#     Differentiable Convex Optimization Layers
#     '''

#     def __init__(self, net, solver=solver, exact_solver=spsolver, lr=1e-1):
#         super().__init__(net, exact_solver, lr)
#         self.solver = solver

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x).squeeze()
#         loss = 0

#         for ii in range(len(y)):
#             sol_hat = solver.shortest_pathsolution(y_hat[ii])
#             # sol_true = solver.shortest_pathsolution(y[ii])
#             loss += (sol_hat).dot(y[ii])
#         return loss/len(y)
