import time
import numpy as np
import pickle
import copy
from tqdm.auto import tqdm
import sys 
from ortools.graph import pywrapgraph
from ortools.linear_solver import pywraplp
import torch

def linearobj(x,v, **params):
    return 

def bmatching(preds, mult=1000, **kwargs):
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

solver = pywraplp.Solver.CreateSolver('GLOP')
# solver.SuppressOutput()

class bmatching_diverse:
    def __init__(self,p=0.25, q=0.25, relaxation=False) -> None:
        self.p, self.q = p,q
        self.relaxation = relaxation
    def solve(self, preds, match_subs,  **kwargs):
        p,q = self.p, self.q
        relaxation = self.relaxation
    
        solver.Clear()
        mult=1000
        cost = -preds.reshape(50,50)*mult
        m = match_subs.reshape(50,50)
        n1 = len(cost)
        n2 = len(cost[0])
        x = {}
        for i in range(n1):
            for j in range(n2):
                x[i,j] = solver.NumVar(0,1,'') if relaxation else solver.IntVar(0,1,'')

        for i in range(n1):
            solver.Add(solver.Sum([x[i, j] for j in range(n2)]) <= 1)

        for j in range(n2):
            solver.Add(solver.Sum([x[i, j] for i in range(n1)]) <= 1)

        # pairing in same field
        pairing_same = []
        allvars = []
        for i in range(n1):
            for j in range(n2):
                pairing_same.append(x[i,j] * m[i,j])
                allvars.append(x[i,j])
        solver.Add(solver.Sum(pairing_same) >= p*solver.Sum(allvars))

        # pairing in distinct field
        pairing_dis = []
        for i in range(n1):
            for j in range(n2):
                pairing_dis.append(x[i,j] * (1-m[i,j]))
        solver.Add(solver.Sum(pairing_dis) >= q*solver.Sum(allvars))

        obj = []
        for i in range(n1):
            for j in range(n2):
                obj.append(cost[i,j] * x[i,j]) 
        solver.Minimize(solver.Sum(obj))

        status = solver.Solve()
        solution = np.zeros((50,50))

        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n1):
                for j in range(n2):
                    solution[i,j] = x[i,j].solution_value()
        #solver.Clear()
        return solution.reshape(-1)

    def get_qpt_matrices(self, match_subs):
        p,q = self.p, self.q

        # we only have G * x <= h
        
        # Matching
        N1 = np.zeros((50,2500))
        N2 = np.zeros_like(N1)
        b1 = np.ones(50)
        b2 = np.ones_like(b1)
        
        for i in range(50):
            rowmask = np.zeros((50,50))
            colmask = np.zeros_like(rowmask)
            rowmask[i,:] = 1 
            colmask[:,i] = 1
            N1[i] = rowmask.flatten()
            N2[i] = colmask.flatten() 
        
        # Similarity constraint
        Sim = p - match_subs 
        bsim = np.zeros(1)
        
        # Diversity constraint 
        Div = q - 1 + match_subs 
        bdiv = np.zeros_like(bsim)

        G = np.vstack((N1, N2, Sim, Div))
        h = np.concatenate((b1, b2, bsim, bdiv))
        A = torch.Tensor().float()
        b = torch.Tensor().float()
        return A,b, torch.from_numpy(G).float(), torch.from_numpy(h).float()




# def get_qpt_matrices(match_subs, p=0.25, q=0.25, **kwargs):
#     # we only have G * x <= h
    
#     # Matching
#     N1 = np.zeros((50,2500))
#     N2 = np.zeros_like(N1)
#     b1 = np.ones(50)
#     b2 = np.ones_like(b1)
    
#     for i in range(50):
#         rowmask = np.zeros((50,50))
#         colmask = np.zeros_like(rowmask)
#         rowmask[i,:] = 1 
#         colmask[:,i] = 1
#         N1[i] = rowmask.flatten()
#         N2[i] = colmask.flatten() 
    
#     # Similarity constraint
#     Sim = p - match_subs 
#     bsim = np.zeros(1)
    
#     # Diversity constraint 
#     Div = q - 1 + match_subs 
#     bdiv = np.zeros_like(bsim)

#     G = np.vstack((N1, N2, Sim, Div))
#     h = np.concatenate((b1, b2, bsim, bdiv))
#     A = None 
#     b = None 
#     return A,b, G,h




def get_cora():
    """
    Get X,y
    """
    # 
    with open('data/cora_data.pickle', 'rb') as f:
        gt, ft, M = pickle.load(f)
    return ft, gt, M

if __name__ == '__main__':
    x,y,m = get_cora()
    params = {'p':0.5,'q':0.5} 
    idx = 15
    p,m = bmatching_diverse(y[idx], m[idx], **params) , m[idx]
    objective_fun=lambda x,v,**params: x @ v
    print("Objective ",objective_fun(p,y[idx]) )
    
    _,_, G,h = get_qpt_matrices(m, **params)
    ineq = G @ p
    print('G: ', G.shape)
    print('h: ', h.shape)
    print('G @ x: ', ineq.shape)
    csat = (ineq - h) <=0
    print('constraints satisfied ?', csat.all())
    print(ineq -h)
    a,b = np.unique(G, axis=1, return_counts=True)
    print('uniques?', a.shape)
    print('any repetition?', (b > 2).any())