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

# def build_graph(x_max, y_max):
#     '''
#     Use This function to Initialize the graph.
#     Bidrectional graph, where each node is connected to its 8 neighbours.
#     '''
#     name_concat = lambda s1,s2: '_'.join([str(s1), str(s2)])
#     E = []
#     N = [name_concat(x, y) for x in range(x_max) for y in range(y_max)]
#     for i in range(x_max):
#         for j in range(y_max):
#             if (( (x_max-1)> i>0) & ( (y_max-1)> j>0)):
#                 x_minus,x_plus, y_minus, y_plus = -1,2,-1,2
#             elif(i==j==0 ):
#                 x_minus,x_plus, y_minus, y_plus = 0,2,0,2
#             elif ((i==0)&(j==y_max-1)):
#                 x_minus,x_plus, y_minus, y_plus = 0,2,-1,1
#             elif ((i==x_max-1)&(j==0)):
#                 x_minus,x_plus, y_minus, y_plus = -1,1,0,2
#             elif (i==0):
#                 x_minus,x_plus, y_minus, y_plus = 0,2,-1,2
#             elif (j==0):
#                 x_minus,x_plus, y_minus, y_plus = -1,2,0,2            
#             elif ( (i== (x_max -1)) & (j== (y_max-1) )):
#                 x_minus,x_plus, y_minus, y_plus = -1,1,-1,1
#             elif ( (i== (x_max -1))):
#                 x_minus,x_plus, y_minus, y_plus = -1,1,-1,2
#             elif ( (j== (y_max -1))):
#                 x_minus,x_plus, y_minus, y_plus = -1,2,-1,1              
                        
#             E.extend([ ( name_concat(i,j), name_concat(i+p,j+q)) for p in range(x_minus,x_plus) 
#                     for q in range(y_minus, y_plus) if ((p!=0)|(q!=0)) ])


#     G =  nx.DiGraph()
#     G.add_nodes_from(N)
#     G.add_edges_from(E)  
#     return G
# class CvxDifflayer(nn.Module):
#     '''
#     Differentiable CVXPY layers
#     Argument to initialize  
#     shape: (,): a tuple indicating the shape of the input image
#     '''
#     def __init__(self, shape ) -> None:
#         super().__init__()
#         x_max, y_max = shape
#         G = build_graph(x_max, y_max)

#         Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
#         Incidence_mat_pos = Incidence_mat.copy()
#         Incidence_mat_pos[Incidence_mat_pos==-1]=0
#         b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
#         b_vector[0] = 1
#         b_vector[-1] = -1

#         N,V = Incidence_mat.shape

#         x = cp.Variable(V)
#         z = cp.Variable(N)
#         A = cp.Parameter((N,V))
#         A_pos = cp.Parameter((N,V))
#         b = cp.Parameter(N)
#         c = cp.Parameter(N)

#         # constraints = [x >= 0, x<=1,z>=0, z<=1,z[-1]==1, A @ x == b, A_pos@x =z]
#         constraints = [x >= 0, x<=1,z>=0, z<=1, 
#         Incidence_mat @ x == b_vector , Incidence_mat_pos@x ==z]
    
#         constraints = [x >= 0, x<=1,z>=0, z<=1, z[-1]==1,
#         Incidence_mat @ x == b_vector , Incidence_mat_pos@x <=z]
#         objective = cp.Minimize(c @ z)

#         problem = cp.Problem(objective, constraints)
#         # self.layer = CvxpyLayer(problem, parameters=[A,A_pos, b,c], variables=[z,x])
#         self.layer = CvxpyLayer(problem, parameters=[c], variables=[z,x])
#         self.Incidence_mat = Incidence_mat
#         self.Incidence_mat_pos = Incidence_mat_pos
#         self.b = b_vector 
#     def forward(self,weights):
#         layer = self.layer
#         Incidence_mat_trch = torch.from_numpy(self.Incidence_mat)
#         Incidence_mat_pos_trch =  torch.from_numpy(self.Incidence_mat_pos)
#         b_trch = torch.from_numpy(self.b)

#         # sol = layer(Incidence_mat_trch ,Incidence_mat_pos_trch,
#         # b_trch, weights.view(weights.shape[-1]*weights.shape[-1]) )

#         sol = layer(weights.view(weights.shape[-1]*weights.shape[-1]) )
#         return sol[0].view(weights.shape[-1],weights.shape[-1])


# from qpth.qp import QPFunction
# class QptDifflayer(nn.Module):
#     def __init__(self, shape, mu=1e-8 ) -> None:
#         super().__init__()
#         x_max, y_max = shape
#         G = build_graph(x_max, y_max)
#         self.mu  = mu

#         Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
#         Incidence_mat_pos = Incidence_mat.copy()
#         Incidence_mat_pos[Incidence_mat_pos==-1]=0
#         b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
#         b_vector[0] = 1
#         b_vector[-1] = -1

#         N,V = Incidence_mat.shape # N is the number of nodes, V is the bumbe rof edges
#         A = np.concatenate(
#         ( np.concatenate(( np.zeros((N,N)), Incidence_mat ),axis=1), ## AX <= b
#             np.concatenate(( -np.ones((N,N)),Incidence_mat_pos ),axis=1)),## A' X <=z
#             axis=0 ).astype(np.float32)
#         b = np.concatenate(( b_vector, np.zeros(N) )).astype(np.float32)
#         A_lb  = np.concatenate((
#              np.concatenate(( -np.eye(N), np.zeros((N,V)) ),axis=1),
#              np.concatenate(( np.zeros((V,N)), -np.eye(V) ),axis=1)),
#              axis=0 ).astype(np.float32)
#         b_lb = np.zeros(N+V).astype(np.float32)
#         # b_lb[0] = -1
#         A_ub  = np.concatenate((
#              np.concatenate(( np.eye(N), np.zeros((N,V)) ),axis=1),
#              np.concatenate(( np.zeros((V,N)), np.eye(V) ),axis=1)),
#              axis=0 ).astype(np.float32)
#         b_ub = np.ones(N+V).astype(np.float32)

#         # A = np.concatenate((A,A_lb, A_ub   ), axis=0).astype(np.float32)
#         # b = np.concatenate(( b, b_lb, b_ub )).astype(np.float32)
#         C = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
#         d = np.concatenate(( b_lb, b_ub )).astype(np.float32)

        
#         self.A, self.b = torch.from_numpy(A),  torch.from_numpy(b)
#         self.C,self.d = torch.from_numpy(C),  torch.from_numpy(d)


        
#         self.N, self.V =N,V
#         self.solver = QPFunction()
                
#     def forward(self,weights):
#         A_trch, b_trch = self.A, self.b 
#         C_trch, d_trch =  self.C, self.d
#         weights_flat = weights.view(weights.shape[-1]*weights.shape[-1])  

#         N, V = self.N, self.V 
#         weights_concat = torch.cat((weights_flat, torch.zeros(V))).float()

#         Q =   self.mu*torch.eye(A_trch.shape[1]).float()

#         sol = self.solver(Q,
#                             weights_concat ,
#                             C_trch,d_trch, 
#                             A_trch, b_trch, 
#                             )
#         # sol = self.solver(Q,
#         #                     weights_concat , 
#         #                     A_trch, b_trch, 
#         #                     torch.tensor(), torch.tensor()
#         #                     )
#         sol = sol[0]

#         return sol[:N].view(weights.shape[-1],weights.shape[-1])


# # from qpthlocal.qp import QPFunction
# # from qpthlocal.qp import QPSolvers
# # from qpthlocal.qp import make_gurobi_model

# # class QptDifflayer(nn.Module):
# #     def __init__(self, shape, mu=1e-5 ) -> None:
# #         super().__init__()
# #         x_max, y_max = shape
# #         G = build_graph(x_max, y_max)
# #         self.mu  = mu

# #         Incidence_mat = nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
# #         Incidence_mat_pos = Incidence_mat.copy()
# #         Incidence_mat_pos[Incidence_mat_pos==-1]=0
# #         b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
# #         b_vector[0] = -1
# #         b_vector[-1] = 1

# #         N,V = Incidence_mat.shape
# #         A = np.concatenate(
# #         ( np.concatenate(( np.zeros((N,N)), Incidence_mat ),axis=1),
# #             np.concatenate(( -np.ones((N,N)),Incidence_mat_pos ),axis=1)),axis=0
# #         ).astype(np.float32)

# #         b = np.concatenate(( b_vector, np.zeros(N) )).astype(np.float32)
# #         self.A, self.b = torch.from_numpy(A),  torch.from_numpy(b)

# #         self.model_params_quad = make_gurobi_model( A, b,
# #             A, b, mu*np.eye(A.shape[1]) )
# #         self.solver = QPFunction(verbose=False, 
# #                         model_params=self.model_params_quad)
                
# #     def forward(self,weights):
# #         A_trch, b_trch = self.A, self.b 
# #         weights_flat = weights.view(weights.shape[-1]*weights.shape[-1])  
# #         TwoN,NplusV = A_trch.shape
# #         N = TwoN//2
# #         V  = (2*(NplusV) - TwoN)//2 
        
# #         weights_concat = torch.cat((weights_flat, torch.zeros(V))).float()

# #         Q =   self.mu*torch.eye(A_trch.shape[1]).float()

# #         sol = self.solver(Q.expand(1, *Q.shape),
# #                              weights_concat , 
# #                             A_trch.expand(1,*A_trch.shape), b_trch.expand(1,*b_trch.shape), 
# #                             torch.Tensor(), torch.Tensor())
# #         sol = sol[0]

# #         return sol[:N].view(weights.shape[-1],weights.shape[-1])


# from intopt.intopt_model import IPOfunc
# class IntoptDifflayer(nn.Module):
#     def __init__(self, shape,thr=1e-8,damping=1e-8 ) -> None:
#         super().__init__()
#         x_max, y_max = shape
#         G = build_graph(x_max, y_max)
#         self.thr, self.damping  = thr, damping

#         Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
#         Incidence_mat_pos = Incidence_mat.copy()
#         Incidence_mat_pos[Incidence_mat_pos==-1]=0
#         b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
#         b_vector[0] = 1
#         b_vector[-1] = -1

#         N,V = Incidence_mat.shape # N is the number of nodes, V is the bumbe rof edges
#         A = np.concatenate(
#         ( np.concatenate(( np.zeros((N,N)), Incidence_mat ),axis=1), ## AX <= b
#             np.concatenate(( -np.ones((N,N)),Incidence_mat_pos ),axis=1)),## A' X <=z
#             axis=0 ).astype(np.float32)
#         b = np.concatenate(( b_vector, np.zeros(N) )).astype(np.float32)
#         A_lb  = np.concatenate((
#              np.concatenate(( -np.eye(N), np.zeros((N,V)) ),axis=1),
#              np.concatenate(( np.zeros((V,N)), -np.eye(V) ),axis=1)),
#              axis=0 ).astype(np.float32)
#         b_lb = np.zeros(N+V).astype(np.float32)
#         # b_lb[0] = -1
#         A_ub  = np.concatenate((
#              np.concatenate(( np.eye(N), np.zeros((N,V)) ),axis=1),
#              np.concatenate(( np.zeros((V,N)), np.eye(V) ),axis=1)),
#              axis=0 ).astype(np.float32)
#         b_ub = np.ones(N+V).astype(np.float32)

#         # A = np.concatenate((A,A_lb, A_ub   ), axis=0).astype(np.float32)
#         # b = np.concatenate(( b, b_lb, b_ub )).astype(np.float32)
#         C = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
#         d = np.concatenate(( b_lb, b_ub )).astype(np.float32)

        
#         self.A, self.b = torch.from_numpy(A),  torch.from_numpy(b)
#         self.C,self.d = torch.from_numpy(C),  torch.from_numpy(d)
        
#         self.N, self.V =N,V

#     def forward(self,weights):
#         A_trch, b_trch = self.A, self.b 
#         C_trch, d_trch =  self.C, self.d
#         weights_flat = weights.view(weights.shape[-1]*weights.shape[-1])  

#         N, V = self.N, self.V 
#         weights_concat = torch.cat((weights_flat, torch.zeros(V))).float()

#         # bounds  = [(1.,1.)]+ [(0,1.)]* (N -2) +[(1.,1.)] +  [(0,1)]* (V)
#         #sol = IPOfunc(A =None,b=None,G=A_trch,h=b_trch,thr=self.thr,damping= self.damping)(weights_concat)
#         # sol = IPOfunc(A =A_trch,b=b_trch,G=C_trch,h= d_trch,thr=self.thr,damping= self.damping)(weights_concat)
#         sol = IPOfunc(A =A_trch,b=b_trch,G=None,h= None,thr=self.thr,damping= self.damping)(weights_concat)

#         return sol[:N].view(weights.shape[-1],weights.shape[-1])



def build_graph(x_max, y_max):
    '''
    Use This function to Initialize the graph.
    Bidrectional graph, where each node is connected to its 8 neighbours.
    '''
    name_concat = lambda *s: '_'.join( list(map(str, s)) )
    E = [  ( name_concat(x,y,'in'), name_concat(x,y,'out')) for x in range(x_max) for y in range(y_max)   ]  
    N = [name_concat(x, y, s) for x in range(x_max) for y in range(y_max) for s in ['in','out']]
    '''
    The goal is to create a directed graph with (x_max*y_max) nodes.
    Each node is connected to its 8 neighbours- (x-1,y), (x-1,y+1),(x,y+1),(x+1,y+1), (x+1,y),(x+1,y-1),
    (x,y-1),(x-1,y-1). Care is taken for node which does not have 8 neighbours. 
    '''
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



            E.extend([ ( name_concat(i,j,'out'), name_concat(i+p,j+q,'in')) for p in range(x_minus,x_plus) 
                    for q in range(y_minus, y_plus) if ((p!=0)|(q!=0)) ])
            # E.extend([ ( name_concat(i+p,j+q), name_concat(i,j) ) for p in range(x_minus,x_plus) 
            #         for q in range(y_minus,y_plus) if ((p!=0)|(q!=0)) ])
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
    def __init__(self, shape,mu=1e-8 ) -> None:
        super().__init__()
        x_max, y_max = shape
        G = build_graph(x_max, y_max)
        self.non_zero_edge_idx = [ i for i,k in enumerate( list(G.edges) ) if "_".join(k[0].split("_", 2)[:2]) == "_".join(k[1].split("_", 2)[:2])]

        Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)

        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = 1
        b_vector[-1] = -1

        self.N, self.V = Incidence_mat.shape
        N, V = self.N, self.V
        x = cp.Variable(V)
        # z = cp.Variable(N)
        # A = cp.Parameter((N,V))
        # A_pos = cp.Parameter((N,V))
        # b = cp.Parameter(N)
        c = cp.Parameter(V)

        # constraints = [x >= 0, x<=1,z>=0, z<=1,z[-1]==1, A @ x == b, A_pos@x =z]
        constraints = [x >= 0, x<=1, Incidence_mat @ x == b_vector ]

        objective = cp.Minimize(c @ x + mu*cp.pnorm(x, p=2))

        problem = cp.Problem(objective, constraints)
        # self.layer = CvxpyLayer(problem, parameters=[A,A_pos, b,c], variables=[z,x])
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
        self.Incidence_mat = Incidence_mat
        self.b = b_vector 
        self.mu = mu
     
    def forward(self,weights):
        layer = self.layer
        N, V = self.N, self.V

        weights_flatten = weights.view(weights.shape[-1]*weights.shape[-1])
        expanded_c = torch.zeros(V)
        expanded_c [self.non_zero_edge_idx ] = weights_flatten
   
        # print(expanded_c[0:10])

        # print(expanded_c[10:20])

        # print(expanded_c[100:120])

        # print(expanded_c[450:480])        

        # sol = layer(Incidence_mat_trch ,Incidence_mat_pos_trch,
        # b_trch, weights.view(weights.shape[-1]*weights.shape[-1]) )

        sol, = layer(expanded_c )
        sol = sol[self.non_zero_edge_idx]
        return sol.view(weights.shape[-1],weights.shape[-1])

from qpth.qp import QPFunction
class QptDifflayer(nn.Module):
    def __init__(self, shape, mu=1e-8 ) -> None:
        super().__init__()
        x_max, y_max = shape
        G = build_graph(x_max, y_max)
        self.non_zero_edge_idx = [ i for i,k in enumerate( list(G.edges) ) if "_".join(k[0].split("_", 2)[:2]) == "_".join(k[1].split("_", 2)[:2])]
        self.mu  = mu

        Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        
        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = 1
        b_vector[-1] = -1

        N,V = Incidence_mat.shape # N is the number of nodes, V is the bumbe rof edges

        A_lb  = -np.eye(V).astype(np.float32)
        b_lb = np.zeros(V).astype(np.float32)
        A_ub  = np.eye(V).astype(np.float32)
        b_ub = np.ones(V).astype(np.float32)

        # A = np.concatenate((A,A_lb, A_ub   ), axis=0).astype(np.float32)
        # b = np.concatenate(( b, b_lb, b_ub )).astype(np.float32)
        C = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
        d = np.concatenate(( b_lb, b_ub )).astype(np.float32)

        
        self.A, self.b = torch.from_numpy(Incidence_mat),  torch.from_numpy(b_vector)
        self.C,self.d = torch.from_numpy(C),  torch.from_numpy(d)


        
        self.N, self.V =N,V
        self.solver = QPFunction()
                
    def forward(self,weights):
        A_trch, b_trch = self.A, self.b 
        C_trch, d_trch =  self.C, self.d
        

        N, V = self.N, self.V 
        weights_flatten = weights.view(weights.shape[-1]*weights.shape[-1])
        expanded_c = torch.zeros(V)
        expanded_c[self.non_zero_edge_idx ] = weights_flatten
        # print(expanded_c[0:10])

        # print(expanded_c[10:20])

        # print(expanded_c[100:120])

        # print(expanded_c[450:480])  
        Q =   self.mu*torch.eye(A_trch.shape[1]).float()

        sol = self.solver(Q,
                            expanded_c ,
                            C_trch,d_trch, 
                            A_trch, b_trch, 
                            )
        # sol = self.solver(Q,
        #                     weights_concat , 
        #                     A_trch, b_trch, 
        #                     torch.tensor(), torch.tensor()
        #                     )
        sol = sol[0][self.non_zero_edge_idx ]

        return sol.view(weights.shape[-1],weights.shape[-1])


from intopt.intopt import intopt
class IntoptDifflayer(nn.Module):
    def __init__(self, shape,thr=1e-8,damping=1e-8, diffKKT = False ) -> None:
        super().__init__()
        self.thr, self.damping  = thr, damping
        x_max, y_max = shape
        G = build_graph(x_max, y_max)
        self.non_zero_edge_idx = [ i for i,k in enumerate( list(G.edges) ) if "_".join(k[0].split("_", 2)[:2]) == "_".join(k[1].split("_", 2)[:2])]
        

        Incidence_mat = -nx.incidence_matrix(G, oriented=True).todense().astype(np.float32)
        
        b_vector  = np.zeros(len(Incidence_mat)).astype(np.float32)
        b_vector[0] = 1
        b_vector[-1] = -1

        N,V = Incidence_mat.shape # N is the number of nodes, V is the bumbe rof edges

        A_lb  = -np.eye(V).astype(np.float32)
        b_lb = np.zeros(V).astype(np.float32)
        A_ub  = np.eye(V).astype(np.float32)
        b_ub = np.ones(V).astype(np.float32)

        # A = np.concatenate((A,A_lb, A_ub   ), axis=0).astype(np.float32)
        # b = np.concatenate(( b, b_lb, b_ub )).astype(np.float32)
        C = np.concatenate((A_lb, A_ub   ), axis=0).astype(np.float32)
        d = np.concatenate(( b_lb, b_ub )).astype(np.float32)

        
        self.A, self.b = torch.from_numpy(Incidence_mat),  torch.from_numpy(b_vector)
        self.C,self.d = torch.from_numpy(C),  torch.from_numpy(d)        
        self.N, self.V =N,V
        self.intoptsolver = intopt( self.A, self.b, None, None, thr= thr, damping=damping, dopresolve=True, diffKKT = diffKKT)

    def forward(self,weights):

        A_trch, b_trch = self.A, self.b 
        C_trch, d_trch =  self.C, self.d
        

        N, V = self.N, self.V 
        weights_flatten = weights.view(weights.shape[-1]*weights.shape[-1])
        expanded_c = torch.zeros(V)
        expanded_c[self.non_zero_edge_idx ] = weights_flatten
        print(expanded_c[0:10])

        print(expanded_c[10:20])

        print(expanded_c[100:120])

        print(expanded_c[450:480])


        # A_trch, b_trch = self.A, self.b 
        # C_trch, d_trch =  self.C, self.d
        # weights_flat = weights.view(weights.shape[-1]*weights.shape[-1])  

        # N, V = self.N, self.V 
        # weights_concat = torch.cat((weights_flat, torch.zeros(V))).float()

        # # bounds  = [(1.,1.)]+ [(0,1.)]* (N -2) +[(1.,1.)] +  [(0,1)]* (V)
        # #sol = IPOfunc(A =None,b=None,G=A_trch,h=b_trch,thr=self.thr,damping= self.damping)(weights_concat)
        # # sol = IPOfunc(A =A_trch,b=b_trch,G=C_trch,h= d_trch,thr=self.thr,damping= self.damping)(weights_concat)

        sol = self.intoptsolver (expanded_c)

        return sol[self.non_zero_edge_idx ].view(weights.shape[-1],weights.shape[-1])
