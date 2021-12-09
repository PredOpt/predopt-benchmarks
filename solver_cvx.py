import numpy as np
import math
import pickle as pkl
import argparse
import gurobipy as gp
import cvxpy as cp
import numpy as np
from solver import solve_markowitz_cvx, get_markowitz_constraints_cvx


n = 50    # number of assets
p = 5    # number of features
tau = 0.1  # noise level parameter

e = np.ones(n)
L = 2*0.0025*tau*np.random.rand(n,4) - 0.0025*tau
COV = np.matmul(L,L.T) + np.ones(n)*(0.01*tau)**2
w_ = e/10
gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )

#m = gp.Model("qp")
#w = m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
c = np.ones(n)  # placeholder
#obj = c @ w
#m.setObjective(obj)
#m.addConstr(e @ w <= 1, "1")
#m.addConstr(w @ COV @ w <= gamma, "2")
#m.optimize()

constraints, variables = get_markowitz_constraints_cvx(n,p,tau,L)
sol = solve_markowitz_cvx(constraints, variables, c)
print("A solution x is")
print(sol)


quit()


x = cp.Variable(n)
constraints = [  cp.quad_form( x,COV ) <= gamma,
                 e @ x <= 1]
prob = cp.Problem(cp.Maximize( c.T @ x ),   # (1/2)*cp.quad_form(x, P) +
                  constraints)
prob.solve()



# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)







"""
# Generate a random non-trivial quadratic program.
m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @ P
q = np.random.randn(n)
G = np.random.randn(n, n)
h = G @ np.random.randn(n)
A = np.random.randn(p, n)
b = np.random.randn(p)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h,
                 cp.quad_form( x, P ) <= h,
                  A @ x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)
"""
