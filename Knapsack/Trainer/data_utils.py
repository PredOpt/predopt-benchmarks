import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
import sklearn
from Trainer.comb_solver import knapsack_solver

class Datawrapper():
    def __init__(self, X,y, sol=None,solver=None):
        assert (sol is not None) or (solver is not None)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) 
        if sol is None:
            sol = []
            for i in range(len(y)):
                sol.append(  solver.solve(y[i]) )
            sol = np.array (sol).astype(np.float32)
        self.sol = sol

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx], self.sol[idx]


class KnapsackDataModule(pl.LightningDataModule):
    def __init__(self,capacity, standardize=True, batch_size=70, generator=None,num_workers=8, seed=0):
        super().__init__()

        data = np.load('Trainer/Data.npz')
        weights = data['weights']
        weights = np.array(weights)
        n_items = len(weights)
        x_train,  x_test, y_train,y_test = data['X_1gtrain'],data['X_1gtest'],data['y_train'],data['y_test']
        x_train = x_train[:,1:]
        x_test = x_test[:,1:]
        if standardize:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        x_train = x_train.reshape(-1,48,x_train.shape[1])
        y_train = y_train.reshape(-1,48)
        x_test = x_test.reshape(-1,48,x_test.shape[1])
        y_test = y_test.reshape(-1,48)
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train,y_test), axis=0)
        x,y = sklearn.utils.shuffle(x,y,random_state=seed)
        x_train, y_train = x[:550], y[:550]
        x_valid, y_valid = x[550:650], y[550:650]
        x_test, y_test = x[650:], y[650:]

        solver = knapsack_solver(weights,capacity= capacity, n_items= len(weights) )

        self.train_df = Datawrapper( x_train,y_train,solver=solver)
        self.valid_df  = Datawrapper( x_valid, y_valid,solver=solver )
        self.test_df = Datawrapper( x_test, y_test,solver=solver )
        self.train_solutions= self.train_df.sol

        self.batch_size = batch_size
        self.generator = generator
        self.num_workers = num_workers

        self.weights, self.n_items = weights, n_items

    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)