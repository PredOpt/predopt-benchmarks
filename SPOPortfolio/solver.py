import sys
sys.path.insert(0, '..')
from predopt_models import Solver
import numpy as np
import math
import torch
import pickle as pkl
import argparse
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp


class PortfolioSolverMarkowitz(Solver):
    def __init__(self, n, p, tau, L):
        self.e = np.ones(n)
        self.COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
        w_ = self.e/10
        self.gamma = 2.25 * np.matmul( np.matmul(w_,self.COV), w_ )
        self.m = gp.Model("qp")
        self.m.setParam('OutputFlag', 0)
        self.w = self.m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

        self.m.addConstr(self.e @ self.w <= 1, "1")
        self.m.addConstr(self.w @ self.COV @ self.w <= self.gamma, "2") 

    def solve(self, c):
        self.m.reset()
        obj = c @ self.w
        self.m.setObjective(obj)
        self.m.optimize()
        return np.array(self.w.X)
    
    def get_constraints_matrix_form(self):
        """This problem has quadratic constraints, not clear what to return here
        """
        return (None, None, None, None)

def get_markowitz(n,p,tau,L):
    e = np.ones(n)
    COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
    w_ = e/10
    gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )

    m = gp.Model("qp")
    m.setParam('OutputFlag', 0)
    w = m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    m.addConstr(e @ w <= 1, "1")
    m.addConstr(w @ COV @ w <= gamma, "2")

    return m, w  # model, variables (need vars to change the objective)


# inputs: model, variables, objective
def solve_markowitz_grb(m,w,c):
    obj = (-c) @ w   # maximize
    m.setObjective(obj)
    m.optimize()

    return np.array(w.X)  # solution







# cvxpy versions
def get_markowitz_constraints_cvx(n,p,tau,L):
    e = np.ones(n)
    COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
    w_ = e/10
    gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )

    x = cp.Variable(n)
    constraints = [  cp.quad_form( x,COV ) <= gamma,
                     e @ x <= 1 ]

    return constraints, x


def solve_markowitz_cvx(constraints,variables,c):
    x = variables

    prob = cp.Problem(cp.Maximize( c.T @ x ),   # (1/2)*cp.quad_form(x, P) +
                      constraints)
    prob.solve()

    return np.array(x.value)





"""
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--p", type=int, default=5)
parser.add_argument("--tau", type=int, default=0.1)
args = parser.parse_args()

n = args.n    # number of assets
p = args.p    # number of features
tau = args.tau  # noise level parameter
L = 2*0.0025*tau*np.random.rand(n,4) - 0.0025*tau

m, w = get_markowitz(n,p,tau,L)

c = np.ones(n)

sol = solve_markowitz(m,w,c)

print("sol = ")
print( sol )
"""









""" # original code in one block
e = np.ones(n)
L = 2*0.0025*tau*np.random.rand(n,4) - 0.0025*tau
COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
w_ = e/10
gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )
# Create a new model
m = gp.Model("qp")
# Create variables
w = m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
c = np.ones(n)  # placeholder
obj = c @ w
m.setObjective(obj)
m.addConstr(e @ w <= 1, "1")
m.addConstr(w @ COV @ w <= gamma, "2")
m.optimize()
print('Obj: %g' % obj.getValue())
"""
