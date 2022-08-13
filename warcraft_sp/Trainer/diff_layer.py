from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from comb_modules.dijkstra import get_solver
from Trainer.utils import maybe_parallelize, shortest_pathsolution

def BlackboxDifflayer( lambda_val, neighbourhood_fn="8-grid"):
    solver = get_solver(neighbourhood_fn)
    class BlackboxDifflayer_cls(torch.autograd.Function):
        # def __init__(ctx, lambda_val, neighbourhood_fn="8-grid"):
        #     ctx.lambda_val = lambda_val
        #     ctx.neighbourhood_fn = neighbourhood_fn
        
        @staticmethod
        def forward(ctx, weights):
            ctx.weights = weights.detach().cpu().numpy()
            # ctx.suggested_tours = np.asarray (maybe_parallelize(solver, arg_list=list(ctx.weights)))
            # return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)
            ctx.suggested_tours = shortest_pathsolution(solver, weights)
            return ctx.suggested_tours
        @staticmethod
        def backward(ctx, grad_output):
            assert grad_output.shape == ctx.suggested_tours.shape
            grad_output_numpy = grad_output.detach().cpu().numpy()
            weights_prime = np.maximum(ctx.weights + lambda_val * grad_output_numpy, 0.0)
            better_paths = np.asarray(maybe_parallelize( solver, arg_list=list(weights_prime)))
            better_paths = torch.from_numpy(better_paths).float().to(grad_output.device)
            gradient = -(ctx.suggested_tours - better_paths) / lambda_val
            return   gradient #torch.from_numpy(gradient).to(grad_output.device)
    return BlackboxDifflayer_cls.apply


def  SPOlayer(  neighbourhood_fn="8-grid"):
    solver = get_solver(neighbourhood_fn)
    class SPOlayer_cls(torch.autograd.Function):
        # def __init__(ctx, lambda_val, neighbourhood_fn="8-grid"):
        #     ctx.lambda_val = lambda_val
        #     ctx.neighbourhood_fn = neighbourhood_fn
        
        @staticmethod
        def forward(ctx, weights, label, true_weights):
            ctx.save_for_backward(weights, label, true_weights)
            ctx.suggested_tours = shortest_pathsolution(solver, weights)
            return ctx.suggested_tours
        @staticmethod
        def backward(ctx, grad_output):
            weights, label, true_weights = ctx.saved_tensors
            spo_tour = shortest_pathsolution(solver, 2*weights - true_weights)
            
            gradient = (label - spo_tour)
            # assert grad_output.shape == ctx.suggested_tours.shape
            # grad_output_numpy = grad_output.detach().cpu().numpy()
            # weights_prime = np.maximum(ctx.weights + lambda_val * grad_output_numpy, 0.0)
            # better_paths = np.asarray(maybe_parallelize( solver, arg_list=list(weights_prime)))
            # better_paths = torch.from_numpy(better_paths).float().to(grad_output.device)
            # gradient = -(ctx.suggested_tours - better_paths) / lambda_val
            return   gradient, None, None #torch.from_numpy(gradient).to(grad_output.device)
    return SPOlayer_cls.apply

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
import networkx as nx

def build_graph(x_max, y_max):
    '''
    Use This function to Initialize the graph.
    Bidrectional graph, where each node is connected to its 8 neighbours.
    '''
    name_concat = lambda s1,s2: '_'.join([str(s1), str(s2)])
    E = []
    N = [name_concat(x, y) for x in range(x_max) for y in range(y_max)]
    for i in range(x_max):
        for j in range(y_max):
            if (( (x_max-1)> i>0) & ( (y_max-1)> j>0)):
                x_minus,x_plus, y_minus, y_plus = -1,2,-1,2
            elif(i==j==0 ):
                x_minus,x_plus, y_minus, y_plus = 0,2,0,2
            elif ((i==0)&(j==y_max-1)):
                x_minus,x_plus, y_minus, y_plus = 0,2,-1,1
            elif ((i==x_max-1)&(j==0)):
                x_minus,x_plus, y_minus, y_plus = -1,1,0,2
            elif (i==0):
                x_minus,x_plus, y_minus, y_plus = 0,2,-1,2
            elif (j==0):
                x_minus,x_plus, y_minus, y_plus = -1,2,0,2            
            elif ( (i== (x_max -1)) & (j== (y_max-1) )):
                x_minus,x_plus, y_minus, y_plus = -1,1,-1,1
            elif ( (i== (x_max -1))):
                x_minus,x_plus, y_minus, y_plus = -1,1,-1,2
            elif ( (j== (y_max -1))):
                x_minus,x_plus, y_minus, y_plus = -1,2,-1,1              
                
            
                    
            E.extend([ ( name_concat(i,j), name_concat(i+p,j+q)) for p in range(x_minus,x_plus) 
                    for q in range(y_minus, y_plus) if ((p!=0)|(q!=0)) ])
            E.extend([ ( name_concat(i+p,j+q), name_concat(i,j) ) for p in range(x_minus,x_plus) 
                    for q in range(y_minus,y_plus) if ((p!=0)|(q!=0)) ])

    G =  nx.DiGraph()
    G.add_nodes_from(N)
    G.add_edges_from(E)  
    return G


class CvxDifflayer(nn.Module):
    '''
    Differentiable CVXPY layers
    Argument to initialize  
    shape: (,): a tuple indicating the shape of the input image
    '''
    def __init__(self, shape ) -> None:
        super().__init__()
        x_max, y_max = shape
        G = build_graph(x_max, y_max)

        Incidence_mat = nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        Incidence_mat_pos = Incidence_mat.copy()
        Incidence_mat_pos[Incidence_mat_pos==-1]=0
        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = -1
        b_vector[-1] = 1

        N,V = Incidence_mat.shape

        x = cp.Variable(V)
        z = cp.Variable(N)
        A = cp.Parameter((N,V))
        A_pos = cp.Parameter((N,V))
        b = cp.Parameter(N)
        c = cp.Parameter(N)

        constraints = [x >= 0, x<=1,z>=0, z<=1, A @ x <= b, A_pos@x <=z]
        objective = cp.Minimize(c @ z)

        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[A,A_pos, b,c], variables=[z,x])
        self.Incidence_mat = Incidence_mat
        self.Incidence_mat_pos = Incidence_mat_pos
        self.b = b_vector 
    def forward(self,weights):
        layer = self.layer
        Incidence_mat_trch = torch.from_numpy(self.Incidence_mat)
        Incidence_mat_pos_trch =  torch.from_numpy(self.Incidence_mat_pos)
        b_trch = torch.from_numpy(self.b)

        sol = layer(Incidence_mat_trch ,Incidence_mat_pos_trch,
        b_trch, weights.view(-1,weights.shape[-1]*weights.shape[-1]) )
        return sol[0].view(-1,weights.shape[-1],weights.shape[-1])

from intopt.intopt_model import IPOfunc
class IntoptDifflayer(nn.Module):
    def __init__(self, shape,thr=0.1,damping=1e-3 ) -> None:
        super().__init__()
        x_max, y_max = shape
        G = build_graph(x_max, y_max)
        self.thr, self.damping  = thr, damping

        Incidence_mat = nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        Incidence_mat_pos = Incidence_mat.copy()
        Incidence_mat_pos[Incidence_mat_pos==-1]=0
        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = -1
        b_vector[-1] = 1

        N,V = Incidence_mat.shape



        A = np.concatenate(
        ( np.concatenate(( np.zeros((N,N)), Incidence_mat ),axis=1),
            np.concatenate(( -np.ones((N,N)),Incidence_mat_pos ),axis=1)),axis=0
        )

        b = np.concatenate((np.zeros(N) ,b_vector ))
        self.A, self.b = torch.from_numpy(A),  torch.from_numpy(b)

    def forward(self,weights):
        weights_flat = weights.view(-1,weights.shape[-1]*weights.shape[-1])
        A,b = self.A, self.b 
        A_trch, b_trch = A, b

        sol = IPOfunc(A =None,b=None,G=A_trch,h=b_trch,thr=self.thr,damping= self.damping)(weights_flat)
        return sol[:len(A)]




class QptDifflayer(nn.Module):
    def __init__(self, shape ) -> None:
        super().__init__()
        x_max, y_max = shape
        G = build_graph(x_max, y_max)
        Incidence_mat = nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        Incidence_mat_pos = Incidence_mat.copy()
        Incidence_mat_pos[Incidence_mat_pos==-1]=0
        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = -1
        b_vector[-1] = 1

        N,V = Incidence_mat.shape
