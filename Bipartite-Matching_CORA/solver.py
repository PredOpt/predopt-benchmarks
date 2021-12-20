
from ortools.graph import pywrapgraph
import numpy as np

class Bipartite_Matching_Solver:
    def __init__(self, n1, n2) -> None:
        self.model = pywrapgraph.LinearSumAssignment()
        self.prec = 1e-5
        self.n1 = n1
        self.n2 = n2 
        


    def solve(self, costs):
        pass

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