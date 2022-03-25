import pandas as pd
from get_energy import get_energy
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn
import gurobipy as gp
(x_train, y_train,x_test,y_test) = get_energy("prices2013.dat")
x_train = x_train[:,1:]
x_test = x_test[:,1:]
x_valid, x_test = x_test[0:2880,:], x_test[2880:,:]
y_valid, y_test = y_test[0:2880], y_test[2880:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,48,x_train.shape[1])
y_train = y_train.reshape(-1,48)
x_test = x_test.reshape(-1,48,x_test.shape[1])
y_test = y_test.reshape(-1,48)
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
x,y = sklearn.utils.shuffle(x,y,random_state=0)
#### Each knapsack Instance cosnists of 48 items
#### Each item is o fUnit weight, so choose a capacity between 1 and 48


weights = [[1 for i in range(48)]]        
weights = np.array(weights)
n_items = 48
capacity = 10

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
### 552 training instances, 177 test instances

def gurobi_knapsack(y, weights,capacity,n_items,relaxed=False):
    vtype = gp.GRB.CONTINUOUS if relaxed else gp.GRB.BINARY
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    x = model.addMVar(shape= n_items, lb=0.0, ub=1.0,vtype=vtype, name="x")
    model.addConstr(weights @ x <= capacity, name="eq")
    model.setObjective(y@x, gp.GRB.MAXIMIZE)
    model.optimize()
    return x.X
###### Let's test with the instances in test set
for y in y_test:
    sol = gurobi_knapsack(y,weights,capacity= capacity,n_items=n_items)
    print("solution",sol)