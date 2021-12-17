import sys
sys.path.insert(0, '..')
import os 
from predopt_models import Datawrapper, TwoStageRegression
import torch.nn as nn
import pandas as pd
import numpy as np 
import pytorch_lightning as pl
from solver import spsolver
from simulate_shortest_path import generate_instance

from torch.utils.data import DataLoader
######################################  Data Reading #########################################
if not os.path.exists('synthetic_path/data_N_100_noise_0.5_deg_1.csv'):
    generate_instance(100, noise=0.5, deg=1)

df = pd.read_csv("synthetic_path/data_N_100_noise_0.5_deg_1.csv")
y = df.iloc[:,3].values
x= df.iloc[:,4:9].values
x =  x.reshape(-1,36,5).astype(np.float32)
y = y.reshape(-1,36).astype(np.float32)
n_samples =  len(x)
n_training = n_samples*9//10
n_test = n_samples - n_training
print("N training",n_training)
x_train, y_train = x[:n_training], y[:n_training]
x_test, y_test = x[n_training:], y[n_training:]
train_df =  Datawrapper( x_train,y_train,spsolver )
test_df =  Datawrapper( x_test,y_test, spsolver)
train_dl = DataLoader(train_df, batch_size= 16)
test_dl = DataLoader(test_df, batch_size= 2)

# ######################################  Two Stage #########################################
if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
    model = TwoStageRegression(net=nn.Linear(5,1), exact_solver=spsolver, lr= 0.01)
    trainer.fit(model, train_dl,test_dl)
    result = trainer.test(test_dataloaders=test_dl)