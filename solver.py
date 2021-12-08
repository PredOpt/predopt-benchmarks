import numpy as np
import math
import pickle as pkl
import argparse
import gurobipy as gp
from gurobipy import GRB


def get_markowitz(n,p,tau,L):
    e = np.ones(n)
    COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
    w_ = e/10
    gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )

    m = gp.Model("qp")
    w = m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    m.addConstr(e @ w <= 1, "1")
    m.addConstr(w @ COV @ w <= gamma, "2")

    return m, w  # model, variables (need vars to change the objective)


# inputs: model, variables, objective
def solve_markowitz(m,w,c):
    obj = c @ w
    m.setObjective(obj)
    m.optimize()

    return np.array(w.X)  # solution




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
