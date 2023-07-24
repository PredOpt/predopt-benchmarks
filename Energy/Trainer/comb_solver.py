import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
from qpth.qp import QPFunction

import gurobipy as gp
from gurobipy import *


def MakeLpMat(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**kwd):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    """
    G1: rows: n_machine * Time; cols: n_task*n_machine* Time
        first T row for machine1, next T: (2T) for machine 2 and so on
        first n_task column for task 1 of machine 1 in time slot 0 then for task 1 machine 2 and so on
    x: decisiion variable-vector of n_task*n_machine* Time. x[  f*(n_task*n_machine* Time)+m*(n_machine* Time)+Time ]=1 if task f starts at time t on machine m.
    A1: To ensure each task is scheduled only once.
    A2: To respect early start time
    A3: To respect late start time
    F: rows:Time , cols: n_task*n_machine* Time, bookkeping for power power use for each time unit
    Code is written assuming nb resources=1
    """
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G1 = torch.zeros((nbMachines*N,nbTasks*nbMachines*N)).float()
    h1 = torch.zeros(nbMachines*N).float()
    F = torch.zeros((N,nbTasks*nbMachines*N)).float()
    for m in Machines:
        for t in range(N):
            ## in all of our problem, we have only one resource
            h1[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G1[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] = U[f][0]
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]

    G2 = torch.eye((nbTasks*nbMachines*N))
    G3 = -1*torch.eye((nbTasks*nbMachines*N))
    h2 = torch.ones(nbTasks*nbMachines*N)
    h3 = torch.zeros(nbTasks*nbMachines*N)

    G = G1 # torch.cat((G1,G2,G3)) 
    h = h1 # torch.cat((h1,h2,h3))
    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        for m in Machines:
            start_index = f*N*nbMachines + m*N # Time 0 for task f machine m
            ## early start time
            A2 [f,start_index:( start_index + E[f]) ] = 1
            ## latest end time
            A3 [f,(start_index+L[f]-D[f]+1):(start_index+N) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

def IconMatrixsolver(A,b,G,h,F,y):
    '''
    A,b,G,h define the problem
    y: the price of each hour
    Multiply y with F to reach the granularity of x
    x is the solution vector for each hour for each machine for each task 
    '''
    n = A.shape[1]
    m = gp.Model("matrix1")
    x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")

    m.addConstr(A @ x == b, name="eq")
    m.addConstr(G @ x <= h, name="ineq")
    c  = np.matmul(F,y).squeeze()
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.optimize()
    if m.status==2:
        return x.X


def ICONSolutionPool(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
    verbose=False,method=-1,**kwd):


    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)


    N = 1440//q

    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)
   
    x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")


    M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
    M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)
    M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr( quicksum( quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                               U[f][r] for f in Tasks) <= MC[m][r]) 
    M.setObjective(0, GRB.MINIMIZE)
    M.setParam('PoolSearchMode', 2)
    M.setParam('PoolSolutions', 100)
#     M = M.presolve()
#     M.update()
    M.optimize()
    
    batch_sol_spos = []

    if M.status in [GRB.Status.OPTIMAL]:
        try:
            for i in range(M.SolCount):
                M.setParam('SolutionNumber', i)
                sol = np.zeros(N)

                task_on = np.zeros( (nbTasks,nbMachines,N) )
                for ((f,m,t),var) in x.items():
                    try:
                        task_on[f,m,t] = var.Xn
                    except AttributeError:
                        raise

                for t in range(N):        
                    sol[t] = np.sum( np.sum(task_on[f,:,max(0,t-D[f]+1):t+1])*P[f] for f in Tasks )  
                sol = sol*q/60 
                batch_sol_spos.append(sol)
            return batch_sol_spos
        except NameError:
                print("\n__________Something wrong_______ \n ")
                raise


def data_reading(filename):
    with open(filename) as f:
        mylist = f.read().splitlines()
    
    q= int(mylist[0])
    nbResources = int(mylist[1])
    nbMachines =int(mylist[2])
    idle = [None]*nbMachines
    up = [None]*nbMachines
    down = [None]*nbMachines
    MC = [None]*nbMachines
    for m in range(nbMachines):
        l = mylist[2*m+3].split()
        idle[m] = int(l[1])
        up[m] = float(l[2])
        down[m] = float(l[3])
        MC[m] = list(map(int, mylist[2*(m+2)].split()))
    lines_read = 2*nbMachines + 2
    nbTasks = int(mylist[lines_read+1])
    U = [None]*nbTasks
    D=  [None]*nbTasks
    E=  [None]*nbTasks
    L=  [None]*nbTasks
    P=  [None]*nbTasks
    for f in range(nbTasks):
        l = mylist[2*f + lines_read+2].split()
        D[f] = int(l[1])
        E[f] = int(l[2])
        L[f] = int(l[3])
        P[f] = float(l[4])
        U[f] = list(map(int, mylist[2*f + lines_read+3].split()))
    return {"nbMachines":nbMachines,
                "nbTasks":nbTasks,"nbResources":nbResources,
                "MC":MC,
                "U":U,
                "D":D,
                "E":E,
                "L":L,
                "P":P,
                "idle":idle,
                "up":up,
                "down":down,
                "q":q}



class SolveICON:
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    def __init__(self,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
        relax=True,
        verbose=False,method=-1,**h):
        self.nbMachines  = nbMachines
        self.nbTasks = nbTasks
        self.nbResources = nbResources
        self.MC = MC
        self.U =  U
        self.D = D
        self.E = E
        self.L = L
        self.P = P
        self.idle = idle
        self.up = up
        self.down = down
        self.q= q
        self.relax = relax
        self.verbose = verbose
        self.method = method

       
        
    def make_model(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        relax = self.relax
        q= self.q
        N = 1440//q

        M = Model("icon")
        if not self.verbose:
            M.setParam('OutputFlag', 0)
        if relax:
            x = M.addVars(Tasks, Machines, range(N), lb=0., ub=1., vtype=GRB.CONTINUOUS, name="x")
        else:
            x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")


        M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
        M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)
        M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

        # capacity requirement
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr( quicksum( quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])   
        # M = M.presolve()
        M.update()
        self.model = M

        self.x = dict()
        for var in M.getVars():
            name = var.varName
            if name.startswith('x['):
                (f,m,t) = map(int, name[2:-1].split(','))
                self.x[(f,m,t)] = var

    def solve(self,price,timelimit=None):
        Model = self.model
        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        q= self.q
        N = 1440//q  

        verbose = self.verbose
        x =  self.x
        nbMachines = self.nbMachines
        nbTasks = self.nbTasks
        nbResources = self.nbResources
        Machines = range(nbMachines)
        Tasks = range(nbTasks)
        Resources = range(nbResources)
        obj_expr = quicksum( [x[(f,m,t)]*sum(price[t:t+D[f]])*P[f]*q/60 
            for f in Tasks for t in range(N-D[f]+1) for m in Machines if (f,m,t) in x] )
        
        Model.setObjective(obj_expr, GRB.MINIMIZE)
        #Model.setObjective( quicksum( (x[(f,m,t)]*P[f]*quicksum([price[t+i] for i in range(D[f])])*q/60) for f in Tasks
        #                for m in Machines for t in range(N-D[f]+1)), GRB.MINIMIZE)
        if timelimit:
            Model.setParam('TimeLimit', timelimit)
        #if relax:
        #    Model = Model.relax()
        Model.setParam('Method', self.method)
        #logging.info("Number of constraints%d",Model.NumConstrs)
        Model.optimize()
        
        solver = np.zeros(N)
        if Model.status in [GRB.Status.OPTIMAL]:
            try:
                #task_on = Model.getAttr('x',x)
                task_on = np.zeros( (nbTasks,nbMachines,N) )
                for ((f,m,t),var) in x.items():
                    try:
                        task_on[f,m,t] = var.X
                    except AttributeError:
                        task_on[f,m,t] = 0.
                        print("AttributeError: b' Unable to retrieve attribute 'X'")
                        print("__________Something WRONG___________________________")


                if verbose:
                    
                    print('\nCost: %g' % Model.objVal)
                    print('\nExecution Time: %f' %Model.Runtime)
                    print("where the variables is one: ",np.argwhere(task_on==1))
                for t in range(N):        
                    solver[t] = sum( np.sum(task_on[f,:,max(0,t-D[f]+1):t+1])*P[f] for f in Tasks ) 
                solver = solver*q/60
                self.model.reset(0)  
                return solver
            except NameError:
                print("\n__________Something wrong_______ \n ")
                # make sure cut is removed! (modifies model)
                self.model.reset(0)
                
                return solver

        elif Model.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
        elif Model.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
        elif Model.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
        else:
            print('Optimization ended with status %d' % Model.status)
        self.model.reset(0)

        return solver
