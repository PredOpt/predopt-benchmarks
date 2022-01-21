import sys
sys.path.insert(0, '..')
from ortools.graph import pywrapgraph
import numpy as np
from predopt_models import Solver
import gurobipy as gp
import torch


def solve_bmatching(preds, mult=1000, **kwargs):
    assignment = pywrapgraph.LinearSumAssignment()
    cost = -preds.reshape(50,50)*mult
    n1 = len(cost)
    n2 = len(cost[0])
    for i in range(n1):
        for j in range(n2):
          assignment.AddArcWithCost(i, j, int(cost[i,j]))
    solve_status = assignment.Solve()
    solution = np.zeros((50,50))
    for i in range(assignment.NumNodes()):
        mate = assignment.RightMate(i)
        solution[i,mate] = 1
    return solution.reshape(-1)

class BipartiteMatchingSolver(Solver):
    def __init__(self, N=50):
        self.N = N
        self.N1 = np.zeros((N,N*N))
        self.N2 = np.zeros_like(self.N1)
        self.b1 = np.ones(N)
        self.b2 = np.ones_like(self.b1)
        self.model = gp.Model()
        self.model.setParam('OutputFlag',0)
        _,_, G,h = self.get_constraints_matrix_form()
        self.x = self.model.addMVar(shape=G.shape[1], vtype=gp.GRB.BINARY, name='x')
        self.model.addConstr(G @ self.x <= h, name="ineq")
    
    def get_constraints_matrix_form(self):
        for i in range(self.N):
            rowmask = np.zeros((self.N,self.N))
            colmask = np.zeros_like(rowmask)
            rowmask[i,:] = 1 
            colmask[:,i] = 1
            self.N1[i] = rowmask.flatten()
            self.N2[i] = colmask.flatten() 
        G = np.vstack((self.N1, self.N2))
        h = np.concatenate((self.b1, self.b2))
        return None, None, G, h

    def solve(self, y: np.ndarray):
        self.model.reset()
        self.model.setObjective(y @ self.x, gp.GRB.MAXIMIZE)
        self.model.optimize()
        return self.x.X

        