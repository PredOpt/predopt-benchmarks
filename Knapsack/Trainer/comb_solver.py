from ortools.linear_solver import pywraplp
import numpy as np
class knapsack_solver:
    def __init__(self, weights,capacity,n_items):
        self.weights=  weights
        self.capacity = capacity
        self.n_items = n_items
        self.make_model()
    def make_model(self):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        x = {}
        for i in range(self.n_items):
            x[i] = solver.BoolVar(f'x_{i}')
        solver.Add( sum(x[i] * self.weights[i] for i in range(self.n_items)) <= self.capacity)
        
       
        self.x  = x
        self.solver = solver
    def solve(self,y):
        y= y.astype(np.float64)
        x = self.x
        solver = self.solver
    
        objective = solver.Objective()
        for i in range(self.n_items):
                objective.SetCoefficient(x[i],y[i])
        objective.SetMaximization()   
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            sol = np.zeros(self.n_items)
            for i in range(self.n_items):
                sol[i]= x[i].solution_value()
            return sol
        else:
            raise Exception("No soluton found")