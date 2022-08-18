import networkx as nx
import gurobipy as gp
import numpy as np

# A = nx.adjacency_matrix(G, weight=None).todense()
# I = np.identity(len(A))

name_concat = lambda s1,s2: '_'.join([str(s1), str(s2)])
def ILP(matrix):
    x_max, y_max = matrix.shape
    print("weight of sink node ",matrix[-1,-1])
    row_sum_constraintmat= np.zeros((x_max, x_max*y_max))
    col_sum_constraintmat= np.zeros((y_max, x_max*y_max))
    for i in range(x_max):
        row_sum_constraintmat[i,i*x_max:((i+1)*x_max)]=1

    for j in range(y_max):
        col_sum_constraintmat[j,np.arange(j,x_max*y_max, y_max)]=1    
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

    A = -nx.incidence_matrix(G, oriented=True).todense()
    A_pos = A.copy()
    A_pos[A_pos==-1]=0

    bigM = 1e18
    

    b =  np.zeros(len(A))
    b[0] = 1
    b[-1] = -1
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    # x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
    # z = model.addMVar(shape=A.shape[0], vtype=gp.GRB.BINARY, name="z")


    x = model.addMVar(shape=A.shape[1],lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x")
    z = model.addMVar(shape=A.shape[0],lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="z")

    # model.addConstr( z[0]==1, name="source")
    # model.addConstr( z[-1]==1, name="sink")
  
    model.addConstr( A@ x == b, name="eq")
    model.addConstr( A_pos@ x ==  z, name="eq")
    '''
    Inequality constraint only for sink nodes, as there is no incoming edge at sink, 
    sink node variable can't be 1 otherwise. 
    '''

    model.setObjective(matrix.flatten() @z, gp.GRB.MINIMIZE)
    model.optimize()

    if model.status==2:
        return z.x.reshape( x_max, y_max )
    else:
        print(model.status)
        model.computeIIS()
        model.write("infreasible_nodeweightedSP.ilp")
        raise Exception("Soluion Not found")