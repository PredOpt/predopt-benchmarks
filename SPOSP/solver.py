import numpy as np
import gurobipy as gp
import networkx as nx
import torch
from predopt_models import Solver
import sys

from numpy.core.defchararray import array
sys.path.insert(0, '..')


class ShortestPathSolver(Solver):
    """
    Solve shortest path problem over a directed graph, using Gurobi

    Args:
        n_vertices: number of vertices
    """

    def __init__(self, n_vertices=25):
        V = range(25)
        E = []

        for i in V:
            if (i + 1) % 5 != 0:
                E.append((i, i + 1))
                if i + 5 < 25:
                    E.append((i, i + 5))
        self.G = nx.DiGraph()
        self.G.add_nodes_from(V)
        self.G.add_edges_from(E)

    def get_constraints_matrix_form(self):
        A = nx.incidence_matrix(self.G, oriented=True).todense()
        b = np.zeros(len(A))
        b[0] = -1
        b[-1] = 1
        return A, b, None, None

    def solve(self, y):
        '''
        y the vector of  edge weight
        '''
        # A = nx.incidence_matrix(self.G,oriented=True).todense()
        # b =  np.zeros(len(A))
        # b[0] = -1
        # b[-1] =1
        A, b, _, _ = self.get_constraints_matrix_form()
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @ x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status == 2:
            return x.x


spsolver = ShortestPathSolver()
