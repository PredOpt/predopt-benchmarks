import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
import sklearn
from Trainer.utils import batch_solve
from Trainer.get_energy import get_energy
from Trainer.comb_solver import SolveICON

class EnergyDatasetWrapper():
    def __init__(self, X,y, sol=None, solver=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        if sol is None:
            sol = batch_solve(solver, y)

        self.sol = np.array(sol).astype(np.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx],self.sol[idx]


class EnergyDataModule(pl.LightningDataModule):
    def __init__(self,param, standardize=True, batch_size= 16, generator=None,num_workers=4, seed=0, relax=False):
        super().__init__()

        x_train, y_train, x_test, y_test = get_energy(fname= 'Trainer/prices2013.dat')


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

        solver = SolveICON(relax=relax, **param)
        solver.make_model()

        self.train_df = EnergyDatasetWrapper( x_train,y_train,solver=solver)
        self.valid_df  = EnergyDatasetWrapper( x_valid, y_valid,solver=solver )
        self.test_df = EnergyDatasetWrapper( x_test, y_test,solver=solver )
        self.train_solutions= self.train_df.sol

        self.batch_size = batch_size
        self.generator = generator
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)