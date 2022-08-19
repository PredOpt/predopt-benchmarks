from Trainer.bipartite import get_cora
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
import torch 
from torch.utils.data import DataLoader 
import tqdm
class CoraDatawrapper():
    def __init__(self, x,y, M,solver, params={'p':0.25, 'q':0.25},  relaxation=False, sols=None, verbose=False):
        self.x = x
        self.y = y
        self.m = M
        if sols is not None:
            self.sols = sols
        else:
            y_iter = range(len(self.y))
            it = tqdm(y_iter) if verbose else y_iter
            self.sols = np.array([solver.solve(self.y[i], self.m[i], relaxation=relaxation, **params) for i in it])
            self.sols = torch.from_numpy(self.sols).float()
        
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.m = torch.from_numpy(self.m).float()
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sols[index], self.m[index]


###################################### Dataloader #########################################

class CoraMatchingDataModule(pl.LightningDataModule):
    def __init__(self,solver,params, generator=None,  normalize=False, batch_size: int = 32, num_workers: int=8):
        super().__init__()
        x, y,m = get_cora()

        x_train, x_test = x[:22], x[22:]
        y_train, y_test = y[:22], y[22:]
        m_train, m_test = m[:22], m[22:]


        self.train_df = CoraDatawrapper( x_train,y_train,m_train,solver,params=params)
        self.valid_df = CoraDatawrapper( x_test,y_test,m_test, solver,params=params)
        self.test_df = CoraDatawrapper( x_test,y_test,m_test,solver, params=params)
        ### As we don't have much data, valid and test dataset are same
        self.batch_size = batch_size
        self.generator =  generator
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=5, num_workers=self.num_workers)