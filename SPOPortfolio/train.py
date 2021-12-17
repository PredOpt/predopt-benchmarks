import sys
sys.path.insert(0, '..')
from predopt_models import Datawrapper, TwoStageRegression
import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from solver import get_markowitz, solve_markowitz, PortfolioSolverMarkowitz
import pytorch_lightning as pl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batsize', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--load_dataset', type=str, default='noname.txt')
args = parser.parse_args()

# c_true are generally the target data
# c_pred are then the model-predicted values
# input and output are expected to be torch tensors
def spo_grad(c_true, c_pred, solver, variables):

    c_spo = (2*c_pred - c_true)
    grad = []
    regret = []
    for i in range(len(c_spo)):    # iterate the batch
        sol_true = solve_markowitz(solver,variables, np.array(c_true[i]))
        sol_spo  = solve_markowitz(solver,variables, np.array(c_spo[i]))
        sol_pred  = solve_markowitz(solver,variables, np.array(c_pred[i]))
        grad.append(  torch.Tensor(sol_spo - sol_true)  )
        regret.append(  torch.dot(c_true, torch.Tensor(sol_true - sol_pred)  ) )   # this is only for diagnostic / results output

    grad = torch.stack( grad )
    regret = torch.stack( regret )
    return grad, regret

# model is the ML/NN
# optimizer is the torch object
# solver is the CO/LP/QP solver
# variables is the solver's variable handles
def train_fwdbwd_spo(model, optimizer, solver, variables, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    grad, regret = spo_grad(c_true, c_pred, solver, variables)
    optimizer.zero_grad()
    c_pred.backward(gradient = grad)
    optimizer.step()

    return regret



loaded = pickle.load(open('portfolio_data.pkl','rb'))

( (n_samples,n,p,tau,deg,B,L,f,COV,gamma) , data_pairs ) = loaded

inputs  = torch.stack(  [ torch.Tensor(a) for (a,b) in data_pairs ]  )
targets = torch.stack(  [ torch.Tensor(b) for (a,b) in data_pairs ]  )

print("inputs.shape = ")
print( inputs.shape )
print("targets.shape = ")
print( targets.shape )

# model input size: p (number of features) -> default 5
# model output size: n (number of assets)  -> default 50
model = nn.Linear(p,n)
optimizer = torch.optim.Adam( model.parameters(), lr=args.lr  )


n_training = n_samples*9//10
n_test = n_samples - n_training
print("N training",n_training)
x_train, y_train = inputs[:n_training], targets[:n_training]
x_test, y_test = inputs[n_training:], targets[n_training:]
dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batsize)
dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=2)

solver, variables = get_markowitz(n,p,tau,L)


if __name__ == '__main__':
    
    # for epoch in range(0,args.epochs):
    #     for batch_idx, (input, target) in enumerate(train_loader):

    #         regret = train_fwdbwd_spo(model, optimizer, solver, variables, input, target)
    #         print("regret = {}".format(regret.mean))

    # with torch.no_grad():
    #     for b_idx, (input, target) in enumerate(test_loader):
    #         regret = train_fwdbwd_spo(model, optimizer, solver, variables, input, target)
    #         print(f'regret ={regret.mean}')

    
    solver = PortfolioSolverMarkowitz(n, p, tau, L)
    train_dl = torch.utils.data.DataLoader(Datawrapper(x_train, y_train, solver), batch_size = args.batsize)
    test_dl = torch.utils.data.DataLoader(Datawrapper(x_test, y_test, solver), batch_size = 2)
    trainer = pl.Trainer(max_epochs=args.epochs,  min_epochs=4)
    model = TwoStageRegression(net=nn.Linear(5,1), exact_solver=solver, lr= 0.01)
    trainer.fit(model, train_dl,test_dl)
    result = trainer.test(test_dataloaders=test_dl)
