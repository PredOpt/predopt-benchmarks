import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from solver import get_markowitz_constraints_cvx, solve_markowitz_cvx
from train_utils import spo_grad_cvx, train_fwdbwd_spo_cvx, BlackboxMarkowitzWrapper, train_fwdbwd_blackbox_cvx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batsize', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--blackbox_lambda', type=float, default=15.0)
parser.add_argument('--load_dataset', type=str, default='noname.txt')
parser.add_argument('--train_mode', type=str, default='spo')
args = parser.parse_args()




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

dataset = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batsize)

constraints, variables = get_markowitz_constraints_cvx(n,p,tau,L)
if args.train_mode = 'blackbox':
    blackbox_layer = BlackboxMarkowitzWrapper(constraints, variables, args.blackbox_lambda)()

for epoch in range(0,args.epochs):
    for batch_idx, (input, target) in enumerate(train_loader):

        if args.train_mode = 'spo':
            regret = train_fwdbwd_spo_cvx(model, optimizer, constraints, variables, input, target)
            print("regret = {}".format(regret.mean))

        elif args.train_mode = 'blackbox':
            regret = train_fwdbwd_blackbox_cvx(model, optimizer, constraints, variables, input, target)
            print("regret = {}".format(regret.mean))
