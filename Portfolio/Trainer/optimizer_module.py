import numpy as np
import torch 
from torch import nn, optim
import torch.nn.functional as F

###################################### Gurobi  Solver #########################################
import gurobipy as gp
from gurobipy import GRB
class gurobi_portfolio_solver:
    '''
    Gurobi solver takes the price as parameter, return the solution of the maximizimization problem
    '''
    def __init__(self,  cov, gamma, n_stocks = 50):
        self.n_stocks = n_stocks
        model = gp.Model("qp")
        model.setParam('OutputFlag', 0)

        x = model.addMVar(shape= n_stocks, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

        model.addConstr(sum(x) <= 1, "1")
        ### Original Model invoves inequality, We once tested  with Equality
        # model.addConstr(sum(x) == 1, "1")

        model.addConstr(x @ cov @ x <= gamma, "2")
        self.model = model
        self.x = x
    def solve(self, price):
        model = self.model
        x =  self.x


        model.setObjective(price@x, gp.GRB.MAXIMIZE)
        model.optimize()

        if model.status==2:
            sol = x.x
            sol[sol < 1e-4] = 0
            return sol
        else:
            raise Exception("Optimal Solution not found")   
    def solution_fromtorch(self, y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.solve( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.solve( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

### Build cvxpy model prototype
class cvxsolver:
    ''' 
    Implementation of QPTL with cvxpylayers and quadratic regularizer
    '''
    def __init__(self,cov, gamma, n_stocks = 50, mu=1e-6,regularizer='quadratic'):
        '''
        regularizer: form of regularizer- either quadratic or entropic
        '''
        self.cov = cov
        self.gamma =  gamma
        self.n_stocks =  n_stocks
        self.mu = mu
        self.regularizer = regularizer
    



        x = cp.Variable(n_stocks)
        constraints = [x >= 0, cp.quad_form( x, cov ) <= gamma, cp.sum(x) <= 1]
        ### Original Model invoves inequality, We once tested  with Equality
        # constraints = [x >= 0, cp.quad_form( x, cov ) <= gamma, cp.sum(x) == 1]

        c = cp.Parameter(n_stocks)

        if self.regularizer=='quadratic':
            objective = cp.Minimize(-c @ x+ self.mu*cp.pnorm(x, p=2))  
        elif self.regularizer=='entropic':
            objective = cp.Minimize(-c @ x -  self.mu*cp.sum(cp.entr(x)) )
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
    def solution(self, y):
              
        sol, = self.layer(y)
        return sol

### We cannot run intopt in this problem, It can't handle quadratic onstriants



