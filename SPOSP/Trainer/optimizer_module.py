import networkx as nx
import numpy as np
import torch 
from torch import nn, optim
import torch.nn.functional as F
from qpth.qp import QPFunction
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
##################################   Ortools Shortest path Solver #########################################
from ortools.linear_solver import pywraplp
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

        solver = pywraplp.Solver.CreateSolver('GLOP')

        x = {}
        
        x = [solver.NumVar(0.0, 1, str(jj)) for jj  in range(A.shape[1])]
        
        constraints = []
        for ii in range(len(A)):
            constraints.append(solver.Constraint(b[ii], b[ii]))
            for jj in range(A.shape[1]):
                constraints[ii].SetCoefficient(x[jj], A[ii,jj])
        
        
        objective = solver.Objective()
        for jj in range(A.shape[1]):
            objective.SetCoefficient(x[jj], float(y[jj]))
        objective.SetMinimization()
        status = solver.Solve()
        # print(status)
        sol = np.zeros(A.shape[1])
        if status ==  pywraplp.Solver.OPTIMAL:
            
            for i, v in enumerate(x):
                sol[i] = v.solution_value()
        else:
                raise Exception("Optimal Solution not found")
        return sol
    def solution_fromtorch(self,y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.shortest_pathsolution( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)
###################################### Gurobi Shortest path Solver #########################################
# import gurobipy as gp
# class shortestpath_solver:
#     def __init__(self,G= G):
#         self.G = G
    
#     def shortest_pathsolution(self, y):
#         '''
#         y: the vector of  edge weight
#         '''
#         A = nx.incidence_matrix(G,oriented=True).todense()
#         b =  np.zeros(len(A))
#         b[0] = -1
#         b[-1] =1
#         model = gp.Model()
#         model.setParam('OutputFlag', 0)
#         x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
#         model.setObjective(y @x, gp.GRB.MINIMIZE)
#         model.addConstr(A @ x == b, name="eq")
#         model.optimize()
#         if model.status==2:
#             return x.x
#         else:
#             raise Exception("Optimal Solution not found")
#     def is_uniquesolution(self, y):
#         '''
#         y: the vector of  edge weight
#         '''
#         A = nx.incidence_matrix(G,oriented=True).todense()
#         b =  np.zeros(len(A))
#         b[0] = -1
#         b[-1] =1
#         model = gp.Model()
#         model.setParam('OutputFlag', 0)
#         x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
#         model.setObjective(y @x, gp.GRB.MINIMIZE)
#         model.addConstr(A @ x == b, name="eq")
#         model.setParam('PoolSearchMode', 2)
#         model.setParam('PoolSolutions', 100)
#         #model.PoolObjBound(obj)
#         model.setParam('PoolGap', 0.0)
#         model.optimize()
#         self.model = model
#         return model.SolCount<=1 

#     def highest_regretsolution(self,y,y_true, minimize=True):
#         mm = 1 if minimize else -1
        
#         if self.is_uniquesolution(y):
#             model = self.model
#             return np.array(model.Xn).astype(np.float32), 0
#         else:
#             model = self.model
#             sols = []
#             for solindex in range(model.SolCount):
#                 model.setParam('SolutionNumber', solindex)
#                 sols.append(model.Xn)  
#             sols = np.array(sols).astype(np.float32)
#             # print(sols.dot(y_true))
#             return sols[np.argmax(sols.dot(y_true)*mm, axis=0)], 1 


#     def solution_fromtorch(self,y_torch):
#         if y_torch.dim()==1:
#             return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
#         else:
#             solutions = []
#             for ii in range(len(y_torch)):
#                 solutions.append(torch.from_numpy(self.shortest_pathsolution( y_torch[ii].detach().numpy())).float())
#             return torch.stack(solutions)
#     def highest_regretsolution_fromtorch(self,y_hat,y_true,minimize=True):
#         if y_hat.dim()==1:
#             sol, nonunique_cnt = self.highest_regretsolution( y_hat.detach().numpy(),
#                      y_true.detach().numpy(),minimize  )
#             return torch.from_numpy(sol).float(), nonunique_cnt
#         else:
#             solutions = []
#             nonunique_cnt =0
#             for ii in range(len(y_hat)):
#                 sol,nn = self.highest_regretsolution( y_hat[ii].detach().numpy(),
#                      y_true[ii].detach().numpy(),minimize )
#                 solutions.append(torch.from_numpy(sol).float())
#                 nonunique_cnt += nn
#             return torch.stack(solutions) , nonunique_cnt      
        
spsolver =  shortestpath_solver()

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
from intopt.intopt_model import IPOfunc
from qpth.qp import QPFunction
# from qpthlocal.qp import QPFunction
# from qpthlocal.qp import QPSolvers
# from qpthlocal.qp import make_gurobi_model

### Build cvxpy model prototype
class cvxsolver:
    def __init__(self,G=G, mu=1e-6,regularizer='quadratic'):
        '''
        regularizer: form of regularizer- either quadratic or entropic
        '''
        self.G = G
        self.mu = mu
        self.regularizer = regularizer
    def make_proto(self):
        #### Maybe we can model a better LP formulation
        G = self.G
        num_nodes, num_edges = G.number_of_nodes(),  G.number_of_edges()
        A = cp.Parameter((num_nodes, num_edges))
        b = cp.Parameter(num_nodes)
        c = cp.Parameter(num_edges)
        x = cp.Variable(num_edges)
        constraints = [x >= 0,x<=1,A @ x == b]
        if self.regularizer=='quadratic':
            objective = cp.Minimize(c @ x+ self.mu*cp.pnorm(x, p=2))  
        elif self.regularizer=='entropic':
            objective = cp.Minimize(c @ x -  self.mu*cp.entr(x))  
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[A, b,c], variables=[x])
    def shortest_pathsolution(self, y):
        self.make_proto()
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1        
        sol, = self.layer(A,b,y)
        return sol
    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())  




class intoptsolver:
    def __init__(self,G=G,thr=1e-8,damping=1e-8):
        self.G = G
        self.thr =thr
        self.damping = damping
    def shortest_pathsolution(self, y):
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1      
        sol = IPOfunc(A,b,G=None,h=None,thr=self.thr,damping= self.damping)(y)
        return sol



class qpsolver:
    def __init__(self,G=G,mu=1e-6):
        self.G = G
        A = nx.incidence_matrix(G,oriented=True).todense().astype(np.float32)
        b =  np.zeros(len(A)).astype(np.float32)
        b[0] = -1
        b[-1] = 1
        self.mu = mu
        G_lb = -1*np.eye(A.shape[1])
        h_lb = np.zeros(A.shape[1])
        G_ub = np.eye(A.shape[1])
        h_ub = np.ones(A.shape[1])
        G_ineq = np.concatenate((G_lb, G_ub)).astype(np.float32)
        h_ineq = np.concatenate((h_lb, h_ub)).astype(np.float32)
        Q =  mu*torch.eye(A.shape[1]).float()

        # G_ineq = G_lb
        # h_ineq = h_lb


        # self.model_params_quad = make_gurobi_model(G_ineq,h_ineq,
        #     A, b, np.zeros((A.shape[1],A.shape[1]))  ) #mu*np.eye(A.shape[1])
        # self.solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
        #                 model_params=self.model_params_quad)

        self.A, self.b,self.G, self.h, self.Q =  torch.from_numpy(A), torch.from_numpy(b),  torch.from_numpy(G_ineq),  torch.from_numpy(h_ineq ),  Q
        self.layer = QPFunction()        

    def shortest_pathsolution(self, y):
        A,b,G,h,  Q = self.A, self.b,self.G, self.h, self.Q
        sol = self.layer(Q,y,G,h,A,b)
        return sol


    #     G = self.G
    #     A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
    #     b =  torch.zeros(len(A))
    #     b[0] = -1
    #     b[-1] = 1      
    #     Q =    self.mu*torch.eye(A.shape[1])
    #     ###########   There are two ways we can set the cosntraints of 0<= x <=1
    #     ########### Either specifying in matrix form, or changing the lb and ub in the qp.py file
    #     ########### Curretnyly We're specifying it in constraint form


    #     G_lb = -1*torch.eye(A.shape[1])
    #     h_lb = torch.zeros(A.shape[1])
    #     G_ub = torch.eye(A.shape[1])
    #     h_ub = torch.ones(A.shape[1])
    #     G_ineq = torch.cat((G_lb,G_ub))
    #     h_ineq = torch.cat((h_lb,h_ub))
    #     # G_ineq = G_lb
    #     # h_ineq = h_lb


    #     sol = self.solver(Q.expand(1, *Q.shape),
    #                         y , 
    #                         G_ineq.expand(1,*G_ineq.shape), h_ineq.expand(1,*h_ineq.shape), 
    #                         A.expand(1, *A.shape),b.expand(1, *b.shape))

    #     return sol.squeeze()
    # # def solution_fromtorch(self,y_torch):
    # #     return self.shortest_pathsolution( y_torch.float())  