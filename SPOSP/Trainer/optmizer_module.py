import networkx as nx
import gurobipy as gp
import numpy as np
import torch 
from torch import nn, optim
import torch.nn.functional as F
###################################### Graph Structure ###################################################
V = range(25)
E = []

for i in V:
    if (i+1)%5 !=0:
        E.append((i,i+1))
    if i+5<25:
            E.append((i,i+5))

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)

###################################### Gurobi Shortest path Solver #########################################
class shortestpath_solver:
    def __init__(self,G= G):
        self.G = G
    
    def shortest_pathsolution(self, y):
        '''
        y: the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status==2:
            return x.x
    def is_uniquesolution(self, y):
        '''
        y: the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 100)
        #model.PoolObjBound(obj)
        model.setParam('PoolGap', 0.0)
        model.optimize()
        self.model = model
        return model.SolCount<=1 

    def highest_regretsolution(self,y,y_true, minimize=True):
        mm = 1 if minimize else -1
        
        if self.is_uniquesolution(y):
            model = self.model
            return np.array(model.Xn).astype(np.float32), 0
        else:
            model = self.model
            sols = []
            for solindex in range(model.SolCount):
                model.setParam('SolutionNumber', solindex)
                sols.append(model.Xn)  
            sols = np.array(sols).astype(np.float32)
            # print(sols.dot(y_true))
            return sols[np.argmax(sols.dot(y_true)*mm, axis=0)], 1 


    def solution_fromtorch(self,y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.shortest_pathsolution( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)
    def highest_regretsolution_fromtorch(self,y_hat,y_true,minimize=True):
        if y_hat.dim()==1:
            sol, nonunique_cnt = self.highest_regretsolution( y_hat.detach().numpy(),
                     y_true.detach().numpy(),minimize  )
            return torch.from_numpy(sol).float(), nonunique_cnt
        else:
            solutions = []
            nonunique_cnt =0
            for ii in range(len(y_hat)):
                sol,nn = self.highest_regretsolution( y_hat[ii].detach().numpy(),
                     y_true[ii].detach().numpy(),minimize )
                solutions.append(torch.from_numpy(sol).float())
                nonunique_cnt += nn
            return torch.stack(solutions) , nonunique_cnt      
        
spsolver =  shortestpath_solver()
