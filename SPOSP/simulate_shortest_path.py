"""
Data generation of shortest path instances as it is done in the article
 Smart “predict, then optimize” (2021). The methodology is as follows:
     
     1. Real model parameters are simulated as a bernoulli(probability = 0.5)
     
     2. Real cost per edge are simulated with the formula 
         c_ij = (1 + 1/math.sqrt(p)*(real+3)**deg )*random.uniform(1-noise, 1+noise)
         
         where p is the number of features of the model, real is the simulated real cost,
         deg controls the misespicification of the linear model by creating a polynomial of 
         higher degree, noise is the half-width perturbation.
   

Function generate_instance receive two parameters:
    K: the amount of instances to generate.
    p: the number of features to generate per instance.
    deg: controls the amount of model misspecification
    noise: random perturbation of the real cost

Function compute_shortest_path(data_file): Solve all the instances in data_file
and store it in a file with a prefix "sol_"

References
Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
"""


import random
import csv
import os
import networkx as nx
import math
import pandas as pd
import numpy as np
import gurobipy as gp

class Gurobi_shortestpath:
    def __init__(self,G):
        self.G = G
    def shortestpath(self,source, target):
        G = self.G
        A = nx.incidence_matrix(G,oriented=True).todense()
        edge_attributes = list(G.edges(data = True))
        c = np.array([e[2]['weight'] for e in edge_attributes])

        b = np.zeros(len(A))
        b [source] = -1
        b[target ]= 1    

        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(c @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status==2:
            return x.x, model.objVal   
    def pathlength(self,x):
        G = self.G
        A = nx.incidence_matrix(G,oriented=True).todense()
        edge_attributes = list(G.edges(data = True))
        c = np.array([e[2]['weight'] for e in edge_attributes])        
        return c.dot(x)
    def traversepath(self,x):
        G = self.G
        A = nx.incidence_matrix(G,oriented=True).todense()
        
        reduced_A = A.T[x.astype(bool)]
        return np.where(reduced_A==-1)[1]
    def multishortestpath(self,source, target):
        # obj = self.shortestpath(source, target)[1]
        G = self.G
        A = nx.incidence_matrix(G,oriented=True).todense()
        edge_attributes = list(G.edges(data = True))
        c = np.array([e[2]['weight'] for e in edge_attributes])
        
        b = np.zeros(len(A))
        b [source] = -1
        b[target ]= 1    


        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(c @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 10)
        #model.PoolObjBound(obj)
        model.setParam('PoolGap', 0.)
        model.optimize()
        return model.SolCount



def bernoulli(p):
    
    if random.random()<=p:
        return 1
    else:
        return 0



def generate_instance(datapoints, p=5, deg = 2, noise = 0.5):
    
    file_output = 'synthetic_path/data_N_{0}_noise_{1}_deg_{2}.csv'.format(datapoints,noise,deg)
    
    #Defining Payoffs matrices


    random.seed(0)
    V = range(25)
    E = []
    
    for i in V:
        if (i+1)%5 !=0:
            E.append((i,i+1))
        if i+5<25:
                E.append((i,i+5))
    
    c = {}
    
    
    ff = open(file_output,'w')
    string=['at{0}'.format(i+1) for i in range(p)]
    att = ','.join(string)
    ff.write('data,node_init,node_term,c,'+att+'\n')
    
    #Create one model per coordinate
    
    beta = {e:[bernoulli(0.5) for k in range(p+1)] for e in E}
    
    for i in range(datapoints):
        #Generate the true model    
        
        x = {}
        
        for (u,v) in E:
            x[u,v] =  [round(random.gauss(0,1),3) for k in range(p)]
            
                
            pred = beta[(u,v)][0]+sum(x[u,v][k]*beta[(u,v)][k+1]   for k in range(p) )
            c[u,v] = round((1 + 1/math.sqrt(p)*(pred+3)**deg )*random.uniform(1-noise, 1+noise),5)
            
            attributes_string = ','.join(  str(x[u,v][k]) for k in range(p)   ) 
            
            
            ff.write('{0},{1},{2},{3},{4}\n'.format(i,u,v,c[u,v],attributes_string))
            
    ff.close()
    
    df = pd.read_csv(file_output)
    data  = df.to_numpy()
    np.save(file_output.replace("csv","npy"), data)
    
    compute_shortest_path(file_output)
    
    


def compute_shortest_path(data_file):
    
    N=[]
    E=[]
    V1=[]
    V2=[]
    c={}
    
    with open(data_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
               
                E.append((int(row['node_init']), int(row['node_term'])))
                V1.append(int(row['node_init']))
                V2.append(int(row['node_term']))

                c[int(row['data']),(int(row['node_init']), int(row['node_term']))] = float(row['c'])
                N.append(int(row['data']))
        
        
    E = list(set(E))

    V = list(set(V1+V2))
    N = list(set(N))
    
    s = min(v for v in V)
    t = max(v for v in V)

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    data_sol = data_file.replace('/','/sol_')
    ff = open(data_sol,'w')
    ff.write('i,z,n_optimal,'+','.join(['w_{0}'.format(e) for e in E])+'\n')

    for i in N:
        for e in E:
            
            G[e[0]][e[1]]['weight'] = c[i,e]
            if i ==0: 
                print(e, c[i,e])
        graph = Gurobi_shortestpath(G)
        n_optimal = graph.multishortestpath(source=s, target=t)

        z = nx.bellman_ford_path_length(G, source=s, target=t, weight='weight')
        path = nx.bellman_ford_path(G, source=s, target=t)
        pathGraph =nx.path_graph(path)
        w = {}
        for e in E:
            if e in pathGraph.edges():
                w[e] = 1
            else:
                w[e] = 0
                
        
        ff.write(','.join([str(i),str(z),str(n_optimal)]+[str(w[e]) for e in E])+'\n')
        if n_optimal >1:
            print( 'Non unique solutuins for datafile {} instance {} edge {}'.format(data_file, i,e) )
        # if i==0:
        #     print('PAth 0', z,path)
        #     print([c[i,e] for e in pathGraph.edges()])
    ff.close()
    
    return 'ok'




degs = [1,2,4,6,8]
ns = [100,1000,5000]
noises = [0,0.5]

SETTINGS = [(n,noise,deg) for n in ns for noise in noises for deg in degs]

for (n,noise,deg) in SETTINGS:
    generate_instance(n,noise=noise, deg = deg)




    
    

    
    



