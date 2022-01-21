import sys
import os
import time
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn, optim
import pytorch_lightning as pl
from train import get_dataloaders
from solver import ShortestpathSolver
from predopt_models import SPO, Blackbox, Datawrapper
# from PO_models import twostage_regression, SPO,Blackbox,DCOL, datawrapper
from torch.utils.data import DataLoader

######################################  Data Reading #########################################
# df = pd.read_csv("synthetic_path/data_N_100_noise_0.5_deg_1.csv")
# y = df.iloc[:,3].values
# x= df.iloc[:,4:9].values
# x =  x.reshape(-1,36,5).astype(np.float32)
# y = y.reshape(-1,36).astype(np.float32)
# n_samples =  len(x)
# n_training = n_samples*9//10
# n_test = n_samples - n_training
# print("N training",n_training)
# x_train, y_train = x[:n_training], y[:n_training]
# x_test, y_test = x[n_training:], y[n_training:]
# train_df =  Datawrapper( x_train,y_train)
# test_df =  Datawrapper( x_test,y_test)
# train_dl = DataLoader(train_df, batch_size= 16)
# test_dl = DataLoader(test_df, batch_size= 2)
# ######################################  Two Stage #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = twostage_regression(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl,test_dl)
# result = trainer.test(dataloaders=test_dl)

# ######################################  SPO #########################################
train_dl, valid_dl, test_dl = get_dataloaders("synthetic_path/data_N_100_noise_0.5_deg_1.csv")
trainer = pl.Trainer(max_epochs= 20,  min_epochs=4,
    logger=TensorBoardLogger(
            save_dir=os.path.dirname(os.path.abspath(__file__)), name=f"runs/BB_toast{str(time.time_ns())[4:]}", version="."
        ),
     )
model = Blackbox(net=nn.Linear(5,1) , solver=ShortestpathSolver(), lr= 0.01)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
# ######################################  Blackbox #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = Blackbox(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl, test_dl)
# result = trainer.test(dataloaders=test_dl)
#####################################  Differentiable Convex Optimization Layers  #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = DCOL(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl, test_dl)
# result = trainer.test(dataloaders=test_dl)
#####################################  QPTL  #########################################
