
import numpy as np
import pytorch_lightning as pl
import torch 
from torch.utils.data import DataLoader
###################################### Wrapper #########################################
class datawrapper():
    def __init__(self, x,y, sol=None, solver= None ):
        self.x = x
        self.y = y
        if sol is None:
            if solver is None:
                raise  Exception("Either Give the solutions or provide a solver!") 
            sol = []
            for i in range(len(y)):
                sol.append(   solver.solve(y[i])   )            
            sol = np.array(sol).astype(np.float32)
            
        self.sol = sol

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.sol[index]


###################################### Dataloader #########################################

class ShortestPathDataModule(pl.LightningDataModule):
    def __init__(self, train_df,valid_df,test_df,generator,  normalize=False, batchsize: int = 32, num_workers: int=4):
        super().__init__()
        self.train_df = train_df
        self.valid_df =  valid_df
        self.test_df = test_df
        self.batchsize = batchsize
        self.generator =  generator
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batchsize,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_df, batch_size=self.batchsize,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=1000, num_workers=self.num_workers)
