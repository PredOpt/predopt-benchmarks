import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

def cora_net(n_features=2866, n_hidden=200, n_layers=2, n_targets=1):
    if n_layers ==1:
        return nn.Sequential(nn.Linear(n_features, n_targets), nn.Sigmoid())
    else:
        layers = []
        # input layer
        layers.append(nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU()
            ))
        # hidden layers
        for _ in range(n_layers -2) :
            layers.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            ))
        # output layer
        layers.append(nn.Sequential(
            nn.Linear(n_hidden, n_targets),
            nn.Sigmoid()
        ))
        return nn.Sequential(*layers)

def cora_normednet(n_features=2866, n_hidden=200, n_layers=2, n_targets=1):
    if n_layers ==1:
        return nn.Sequential(nn.Linear(n_features, n_targets), nn.Sigmoid())
    else:
        layers = []
        # input layer
        layers.append(nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),nn.BatchNorm1d(2500)
            ))
        # hidden layers
        for _ in range(n_layers -2) :
            layers.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),nn.BatchNorm1d(2500)
            ))
        # output layer
        layers.append(nn.Sequential(
            nn.Linear(n_hidden, n_targets)
            # nn.Sigmoid()
        ))
        return nn.Sequential(*layers)

def cora_nosigmoidnet(n_features=2866, n_hidden=200, n_layers=2, n_targets=1):
    if n_layers ==1:
        return nn.Sequential(nn.Linear(n_features, n_targets), nn.Sigmoid())
    else:
        layers = []
        # input layer
        layers.append(nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU()
            ))
        # hidden layers
        for _ in range(n_layers -2) :
            layers.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            ))
        # output layer
        layers.append(nn.Sequential(
            nn.Linear(n_hidden, n_targets)
        ))
        return nn.Sequential(*layers)