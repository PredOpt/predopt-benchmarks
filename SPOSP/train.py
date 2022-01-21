import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os 
from predopt_models import Datawrapper
import torch.nn as nn
import pandas as pd
import numpy as np 
from solver import spsolver
from simulate_shortest_path import generate_instance

from torch.utils.data import DataLoader

def split(a):
        return np.split(a, (int(.8*n_samples),int(.9*n_samples)) )

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

def get_dataloaders(data_path):
    df = pd.read_csv(data_path)
    y = df.iloc[:,3].values
    x= df.iloc[:,4:9].values
    x =  x.reshape(-1,36,5).astype(np.float32)
    y = y.reshape(-1,36).astype(np.float32)
    n_samples =  len(x)
    n_training = n_samples*8//10
    n_test = n_samples - n_training
    print("N training",n_training)
    # x_train, y_train = x[:n_training], y[:n_training]
    # x_test, y_test = x[n_training:], y[n_training:]
    
    x_train, x_valid, x_test = split(x)
    y_train, y_valid, y_test = split(y)
    train_df =  Datawrapper( x_train,y_train,spsolver )
    valid_df = Datawrapper(x_valid, y_valid, spsolver)
    test_df =  Datawrapper( x_test,y_test, spsolver)
    train_dl = DataLoader(train_df, batch_size= 16)
    valid_dl = DataLoader(valid_df, batch_size=2)
    test_dl = DataLoader(test_df, batch_size= 2)
    return train_dl, valid_dl, test_dl
