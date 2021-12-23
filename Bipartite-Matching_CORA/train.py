import sys 
sys.path.insert(0, '..')
from predopt_models import Datawrapper, TwoStageRegression
import numpy as np
import torch
import os 
from solver import BipartiteMatchingSolver
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl

assert os.path.exists('data'), "run python make_cora_dataset.py first!"
x = torch.from_numpy(np.load('data/features.npy')).float()
y = torch.from_numpy(np.load('data/true_cost.npy')).float()

x_train, x_test = x[:22], x[22:]
x_train, x_valid = x[:19], x[19:]
y_train, y_test = y[:22], y[22:]
y_train, y_valid = y[:19], y[19:]

bmsolver = BipartiteMatchingSolver()
train_data = Datawrapper(x_train, y_train, bmsolver)
valid_data = Datawrapper(x_valid, y_valid, bmsolver)
test_data = Datawrapper(x_test, y_test, bmsolver)
train_dl = DataLoader(train_data, batch_size=5)
valid_dl = DataLoader(valid_data, batch_size=2)
test_dl = DataLoader(test_data, batch_size=2)

def get_dataloaders(data_path):
    assert os.path.exists(data_path), "run python make_cora_dataset.py first!"
    x = torch.from_numpy(np.load(os.path.join(data_path,'features.npy'))).float()
    y = torch.from_numpy(np.load(os.path.join(data_path,'true_cost.npy'))).float()

    x_train, x_test = x[:22], x[22:]
    x_train, x_valid = x[:19], x[19:]
    y_train, y_test = y[:22], y[22:]
    y_train, y_valid = y[:19], y[19:]

    bmsolver = BipartiteMatchingSolver()
    train_data = Datawrapper(x_train, y_train, bmsolver)
    valid_data = Datawrapper(x_valid, y_valid, bmsolver)
    test_data = Datawrapper(x_test, y_test, bmsolver)
    train_dl = DataLoader(train_data, batch_size=5)
    valid_dl = DataLoader(valid_data, batch_size=2)
    test_dl = DataLoader(test_data, batch_size=2)
    return train_dl, valid_dl, test_dl

def make_cora_net(n_features=2866, n_hidden=200, n_layers=2, n_targets=1):
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

if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs= 5,  min_epochs=4)
    model = TwoStageRegression(net=make_cora_net(), solver=bmsolver, lr= 0.01, twostage_criterion=nn.MSELoss(reduction='mean'))
    trainer.fit(model, train_dl,test_dl)
    result = trainer.test(test_dataloaders=test_dl)

