from ortools.linear_solver import pywraplp
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
from qpth.qp import QPFunction


class knapsack_solver:
    def __init__(self, weights,capacity,n_items):
        self.weights=  weights
        self.capacity = capacity
        self.n_items = n_items
        self.make_model()
    def make_model(self):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        x = {}
        for i in range(self.n_items):
            x[i] = solver.BoolVar(f'x_{i}')
        solver.Add( sum(x[i] * self.weights[i] for i in range(self.n_items)) <= self.capacity)
        
       
        self.x  = x
        self.solver = solver
    def solve(self,y):
        y= y.astype(np.float64)
        x = self.x
        solver = self.solver
    
        objective = solver.Objective()
        for i in range(self.n_items):
                objective.SetCoefficient(x[i],y[i])
        objective.SetMaximization()   
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            sol = np.zeros(self.n_items)
            for i in range(self.n_items):
                sol[i]= x[i].solution_value()
            return sol
        else:
            raise Exception("No soluton found")

class cvx_knapsack_solver(nn.Module):
    def __init__(self, weights,capacity,n_items, mu=1.):
        super().__init__()
        self.weights=  weights
        self.capacity = capacity
        self.n_items = n_items  
        A = weights.reshape(1,-1).astype(np.float32)
        b = capacity
        x = cp.Variable(n_items)
        c = cp.Parameter(n_items)
        constraints = [x >= 0,x<=1,A @ x <= b]  
        objective = cp.Maximize(c @ x - mu*cp.pnorm(x, p=2))  #cp.pnorm(A @ x - b, p=1)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
    def forward(self,costs):
        sol, = self.layer(costs)

        return sol


class qpt_knapsack_solver(nn.Module):
    def __init__(self, weights,capacity,n_items, mu=1.):
        super().__init__()
        self.weights=  weights
        self.capacity = capacity
        self.n_items = n_items  
        A = weights.reshape(1,-1).astype(np.float32)
        b = np.array([capacity]).astype(np.float32)
        A_lb  = -np.eye(n_items).astype(np.float32)
        b_lb = np.zeros(n_items).astype(np.float32)
        A_ub  = np.eye(n_items).astype(np.float32)
        b_ub = np.ones(n_items).astype(np.float32)

        G = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
        h = np.concatenate(( b_lb, b_ub )).astype(np.float32)
        Q =  mu*torch.eye(n_items).float()
        self.A, self.b,self.G, self.h, self.Q =  torch.from_numpy(A), torch.from_numpy(b),  torch.from_numpy(G),  torch.from_numpy(h),  Q
        self.layer = QPFunction()
    def forward(self,costs):
        A,b,G,h,  Q = self.A, self.b,self.G, self.h, self.Q
        sol = self.layer(Q,-costs,G,h,A,b)
        return sol
from intopt.intopt import intopt
class intopt_knapsack_solver(nn.Module):
    def __init__(self, weights,capacity,n_items, thr=0.1,damping=1e-3, diffKKT = False, dopresolve = True,):
        super().__init__()
        self.weights=  weights
        self.capacity = capacity
        self.n_items = n_items  
        A = weights.reshape(1,-1).astype(np.float32)
        b = np.array([capacity]).astype(np.float32)
        A_lb  = -np.eye(n_items).astype(np.float32)
        b_lb = np.zeros(n_items).astype(np.float32)
        A_ub  = np.eye(n_items).astype(np.float32)
        b_ub = np.ones(n_items).astype(np.float32)

        # G = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
        # h = np.concatenate(( b_lb, b_ub )).astype(np.float32)
        self.A, self.b,self.G, self.h =  torch.from_numpy(A), torch.from_numpy(b),  torch.from_numpy(A_ub),  torch.from_numpy(b_ub)
        self.thr =thr
        self.damping = damping
        self.layer = intopt(self.A, self.b,self.G, self.h, thr, damping, diffKKT, dopresolve)

    def forward(self,costs):
        return self.layer(-costs)

        # sol = [self.layer(-cost) for cost in costs]




        # return torch.stack(sol)
            