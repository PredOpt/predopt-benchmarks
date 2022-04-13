import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import networkx as nx
import gurobipy as gp
import perturbations
from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution
import fenchel_young as fy
###################################### Graph Structure ###################################################
V = range(25)
E = []

for i in V:
    if (i+1)%5 !=0:
        E.append((i,i+1))
    if i+5<25:
            E.append((i,i+5))

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)

###################################### Gurobi Shortest path Solver #########################################
class shortestpath_solver:
    def __init__(self,G= G):
        self.G = G
    
    def shortest_pathsolution(self, y):
        '''
        y the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status==2:
            return x.x
    def solution_fromtorch(self,y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.shortest_pathsolution( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)
        
spsolver =  shortestpath_solver()
###################################### Wrapper #########################################
class datawrapper():
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]



######################################  Define the loss functions ###################################### 
def SPOLoss(solver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target):
       
            sol_hat = solver.solution_fromtorch(input)
            sol_spo = solver.solution_fromtorch(2*input - target)
            sol_true = solver.solution_fromtorch(target)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(target)/( sol_true.dot(target) ) # changed to per cent regret

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return mm*(sol_true - sol_spo), None
            
    return SPOLoss_cls.apply
def BlackboxLoss(solver,mu=0.1, minimize=True):
    mm = 1 if minimize else -1
    class BlackboxLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target):
            sol_hat = solver.solution_fromtorch(input)
            sol_perturbed = solver.solution_fromtorch(input + mu* target)
            sol_true = solver.solution_fromtorch(target)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(target)/( sol_true.dot(target) ) # changed to per cent regret

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -mm*(sol_hat - sol_perturbed)/mu, None
            
    return BlackboxLoss_cls.apply


###################################### We will use the SPO forward pass to compute regret ######################################
def regret_fn(solver, y_hat,y, minimize= True):  
    regret_list = []
    for ii in range(len(y)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y[ii]) )
    return torch.mean( torch.tensor(regret_list ))

def regret_aslist(solver, y_hat,y, minimize= True):  
    regret_list = []
    for ii in range(len(y)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y[ii]).item() )
    return np.array(regret_list)
###################################### Regression Model based on MSE loss #########################################
class twostage_regression(pl.LightningModule):
    def __init__(self,net,exact_solver = spsolver, lr=1e-1, l1_weight=0.1, seed=20):
        super().__init__()
        pl.seed_everything(seed)
        # Using seed for reproducibility
        self.net =  net
        self.lr = lr
        self.l1_weight = l1_weight
        self.exact_solver = exact_solver
        # self.batch_size = batch_size
        self.save_hyperparameters("lr",'l1_weight')
    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        loss = criterion(y_hat,y)  + l1penalty * self.l1_weight
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y = batch
        y_hat =  self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_loss =  regret_fn(self.exact_solver, y_hat,y) 
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return {"val_mse":mseloss, "val_regret":regret_loss}
    def validation_epoch_end(self, outputs):
        avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        self.log("ptl/val_regret", avg_regret)
        self.log("ptl/val_mse", avg_mse)
        # self.log("ptl/val_accuracy", avg_acc)
    def test_step(self, batch, batch_idx):
        # same as validation step
        return self.validation_step(batch, batch_idx)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

###################################### SPO and Blackbox Model #########################################


class SPO(twostage_regression):
    def __init__(self,net,solver= spsolver,exact_solver = spsolver,lr=1e-1, l1_weight=0.1, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.solver = solver
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        for ii in range(len(y)):
            loss += SPOLoss(self.solver)(y_hat[ii],y[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss

class Blackbox(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,lr=1e-1,mu =0.1,l1_weight=0.1, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.mu = mu
        self.solver = solver
        self.save_hyperparameters("lr","mu")
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            loss += BlackboxLoss(self.solver,self.mu)(y_hat[ii],y[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss    

###################################### This approach use it's own solver #########################################
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

### Build cvxpy modle prototype
class cvxsolver:
    def __init__(self,G=G):
        self.G = G
    def make_proto(self):
        #### Maybe we can model a better LP formulation
        G = self.G
        num_nodes, num_edges = G.number_of_nodes(),  G.number_of_edges()
        A = cp.Parameter((num_nodes, num_edges))
        b = cp.Parameter(num_nodes)
        c = cp.Parameter(num_edges)
        x = cp.Variable(num_edges)
        constraints = [x >= 0,x<=1,A @ x == b]
        objective = cp.Minimize(c @ x)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[A, b,c], variables=[x])
    def shortest_pathsolution(self, y):
        self.make_proto()
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] =1        
        sol, = self.layer(A,b,y)
        return sol
    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())         



class DCOL(twostage_regression):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.solver = cvxsolver()
    def training_step(self, batch, batch_idx):
        solver = self.solver
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            sol_hat = solver.shortest_pathsolution(y_hat[ii])
            ### The loss is regret but c.dot(y) is constant so need not to be considered
            loss +=  (sol_hat ).dot(y[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss    



from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model

class qpsolver:
    def __init__(self,G=G,mu=1e-6):
        self.G = G
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        self.mu = mu
        G_lb = -1*np.eye(A.shape[1])
        h_lb = np.zeros(A.shape[1])
        G_ub = 1*np.eye(A.shape[1])
        h_ub = np.ones(A.shape[1])
        G_ineq = np.concatenate((G_lb, G_ub))
        h_ineq = np.concatenate((h_lb, h_ub))


        self.model_params_quad = make_gurobi_model(G_ineq,h_ineq,
            A, b,  mu*np.eye(A.shape[1]))
        self.solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
                        model_params=self.model_params_quad)
    def shortest_pathsolution(self, y):
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1      
        Q =    self.mu*torch.eye(A.shape[1])
        ###########   There are two ways we can set the cosntraints of 0<= x <=1
        ########### Either specifying in matrix form, or changing the lb and ub in the qp.py file
        ########### Curretnyly We're specifying it in connstraint form


        G_lb = -1*torch.eye(A.shape[1])
        h_lb = torch.zeros(A.shape[1])
        G_ub = torch.eye(A.shape[1])
        h_ub = torch.ones(A.shape[1])
        G_ineq = torch.cat((G_lb,G_ub))
        h_ineq = torch.cat((h_lb,h_ub))


        sol = self.solver(Q.expand(1, *Q.shape),
                            y , G_ineq.expand(1,*G_ineq.shape), h_ineq.expand(1,*h_ineq.shape), 
                            A.expand(1, *A.shape),b.expand(1, *b.shape))
        return sol
    # def solution_fromtorch(self,y_torch):
    #     return self.shortest_pathsolution( y_torch.float())  


class QPTL(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1, seed=20,mu=0.1):
        self.solver  = qpsolver(mu=mu)

        super().__init__(net,exact_solver , lr, l1_weight, seed)   

from intopt_model import IPOfunc

class intoptsolver:
    def __init__(self,G=G,thr=0.1,damping=1e-3,):
        self.G = G
        self.thr =thr
        self.damping = damping
    def shortest_pathsolution(self, y):
        G = self.G
        A = torch.from_numpy((nx.incidence_matrix(G,oriented=True).todense())).float()  
        b =  torch.zeros(len(A))
        b[0] = -1
        b[-1] = 1      
        sol = IPOfunc(A,b,G=None,h=None,thr=self.thr,damping= self.damping)(y)
        return sol
    
class IntOpt(DCOL):
    '''
    Implementation of
    Differentiable Convex Optimization Layers
    '''
    def __init__(self,net,exact_solver = spsolver,thr=0.1,damping=1e-3,lr=1e-1, l1_weight=0.1, seed=20):
        self.solver  = intoptsolver(thr=thr,damping=damping)

        super().__init__(net,exact_solver , lr, l1_weight, seed)  



###################################### I-MLE #########################################
########## Code adapted from https://github.com/uclnlp/torch-imle/blob/main/annotation-cli.py ###########################



class IMLE(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,k=5,nb_iterations=100,nb_samples=1, 
            lr=1e-1,l1_weight=0.1,input_noise_temperature=1.0, target_noise_temperature=1.0,seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.solver = solver
        self.k = k
        self.nb_iterations = nb_iterations
        self.nb_samples = nb_samples
        self.target_noise_temperature = target_noise_temperature
        self.input_noise_temperature = input_noise_temperature

        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0

        # input_noise_temperature = 1.0
        # target_noise_temperature = 1.0

        target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
        noise_distribution = SumOfGammaNoiseDistribution(k= self.k, nb_iterations=self.nb_iterations)

        @imle(target_distribution=target_distribution,
                noise_distribution=noise_distribution,
                input_noise_temperature=self.input_noise_temperature,
                target_noise_temperature=self.target_noise_temperature,
                nb_samples=self.nb_samples)
        def imle_solver(y):
            #     I-MLE assumes that the solver solves a maximisation problem, but here the `solver` function solves
            # a minimisation problem, so we flip the sign twice. Feed negative cost coefficient to imle_solver and then 
            # flip it again to feed the actual cost to the solver
            return spsolver.solution_fromtorch(-y)

        ########### Also the forward pass returns the solution of the perturbed cost, which is bit strange
        ###########
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        for ii in range(len(y)):
            sol_hat = imle_solver(-y_hat[ii].unsqueeze(0)) # Feed neagtive cost coefficient
            loss +=  (sol_hat*y[ii]).mean()
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss
###################################### Differentiable Perturbed Optimizer #########################################

class DPO(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,lr=1e-1,l1_weight=0.1, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.solver = solver
        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        @perturbations.perturbed(num_samples=10, sigma=0.1, noise='gumbel',batched = False)
        def dpo_solver(y):
            return spsolver.solution_fromtorch(-y)

        for ii in range(len(y)):
            sol_hat = dpo_solver(-y_hat[ii]) # Feed neagtive cost coefficient
            loss +=  ( sol_hat  ).dot(y[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss




################################ Implementation of a Fenchel-Young loss using perturbation techniques #########################################

class FenchelYoung(twostage_regression):
    def __init__(self,net,solver=spsolver,exact_solver = spsolver,lr=1e-1, l1_weight=0.1, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight, seed)
        self.solver = solver
        self.save_hyperparameters("lr")
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        loss = 0
        

        
        def fy_solver(y):
            return spsolver.solution_fromtorch(y)
        ############# Define the Loss functions, we can set maximization to be false

        criterion = fy.FenchelYoungLoss(fy_solver, num_samples=10, sigma=0.1,maximize = False, batched=False)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            loss +=  criterion(y_hat[ii],y[ii])

        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_loss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        return training_loss 
################################ Noise Contrastive Estimation ################################
def batch_solve(solver, y,relaxation =False):
    sol = []
    for i in range(len(y)):
        sol.append(   solver.solution_fromtorch(y[i]).reshape(1,-1)   )
    return torch.cat(sol,0)
def MAP(sol,y,solpool,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    solpool is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, solpool.shape)
    for ii in range(len(y)):
        loss += torch.max(((sol[ii] - solpool )*(mm*y[ii]  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat 
    return MAP(sol,y,solpool,minimize)
def MAP_hatc_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat - y_true
    return MAP(sol,y,solpool,minimize)

def NCE(sol,y,solpool,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    solpool is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, solpool.shape)
    for ii in range(len(y)):
        loss += torch.mean(((sol[ii] - solpool )*(mm*y[ii]  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
    sol = batch_solve( spsolver, y_hat,relaxation =False)
    y = y_hat 
    return NCE(sol,y,solpool,minimize)
def NCE_hatc_c(y_hat,y_true,solpool,minimize=True,*wd,**kwd):
    sol = batch_solve(spsolver, y_hat,relaxation =False)
    y = y_hat - y_true
    return NCE(sol,y,solpool,minimize)



def growpool_fn(solver, solpool, y_hat):
    '''
    solpool is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol = batch_solve(solver, y_hat,relaxation =False).detach().numpy()
    solpool_np = solpool.detach().numpy()
    solpool_np = np.unique(np.append(solpool_np,sol,axis=0),axis=0)
    # torch has no unique function, so we have to do this
    return torch.from_numpy(solpool_np).float()
class SemanticPO(twostage_regression):
    def __init__(self,loss_fn,solpool, net,solver= spsolver,exact_solver = spsolver,growth=0.1,margin=0.,lr=1e-1):
        super().__init__(net,exact_solver,lr)
        # self.save_hyperparameters()
        self.loss_fn = loss_fn
        solpool_np = solpool.detach().numpy()
        solpool_np = np.unique(solpool_np,axis=0)
        # torch has no unique function, so we have to do this
        self.solpool = torch.from_numpy(solpool_np).float()
        self.growpool_fn = growpool_fn
        self.growth = growth
        self.margin = margin
        self.solver = solver
        self.save_hyperparameters("lr","growth","margin")
    
 
    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.solpool)==0:
            #### We have to use seed
            #### Also we are solving for all instances of a batch
            self.solpool = self.growpool_fn(self.solver, self.solpool, y_hat)

  
        loss = self.loss_fn(y_hat,y,self.solpool, self.margin)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss