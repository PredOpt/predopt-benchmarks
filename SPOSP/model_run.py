import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn, optim
import pytorch_lightning as pl
from PO_models import twostage_regression, SPO,Blackbox,DCOL, datawrapper
from torch.utils.data import DataLoader

######################################  Data Reading #########################################
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
train_df =  datawrapper( x_train,y_train)
test_df =  datawrapper( x_test,y_test)
train_dl = DataLoader(train_df, batch_size= 16)
test_dl = DataLoader(test_df, batch_size= 2)
# ######################################  Two Stage #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = twostage_regression(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl,test_dl)
# result = trainer.test(dataloaders=test_dl)

# ######################################  SPO #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = SPO(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl, test_dl)
# result = trainer.test(dataloaders=test_dl)
# ######################################  Blackbox #########################################
# trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
# model = Blackbox(net=nn.Linear(5,1) ,lr= 0.01)
# trainer.fit(model, train_dl, test_dl)
# result = trainer.test(dataloaders=test_dl)
#####################################  Differentiable Convex Optimization Layers  #########################################
trainer = pl.Trainer(max_epochs= 20,  min_epochs=4)
model = DCOL(net=nn.Linear(5,1) ,lr= 0.01)
trainer.fit(model, train_dl, test_dl)
result = trainer.test(dataloaders=test_dl)
#####################################  QPTL  #########################################
