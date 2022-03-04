import sys
import os
import time
from pytorch_lightning import callbacks
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn
import pytorch_lightning as pl
from train import get_dataloaders, make_cora_net
from solver import BipartiteMatchingPool, BipartiteMatchingSolver
from predopt_models import SPO, Blackbox, NCECache, QPTL, TwoStageRegression, SPOTieBreak
from torch.utils.data import DataLoader

######################################  Data Reading #########################################
train_dl, valid_dl, test_dl = get_dataloaders(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
unique_id = str(time.time_ns())[4:]

def make_trainer(name):
    return  pl.Trainer(max_epochs= 20, min_epochs=1, 
    logger=TensorBoardLogger(
            save_dir=os.path.dirname(os.path.abspath(__file__)), name=f"runs/{name}_{unique_id}", version="."
        ),
    callbacks = [EarlyStopping(monitor="val_regret_epoch", patience=2)]
     )
trainers = {}
for name in ['2S-MSE', '2S-CE', 'SPO', 'NCE100', 'NCE10', 'BB', 'QPTL']:
    trainers[name] = make_trainer(name)
res = []

cache = []
for batch in train_dl:
    _,_, sols = batch 
    cache.append(sols)

hparams = {
    # SPO:{
    #     'lr':1e-3
    # },
    # Blackbox:{
    #     'lr':5e-4,
    #     'mu':0.1,
    # },
    # NCECache:{
    #     'lr':5e-3,
    #     'variant':4,
    #     'cache_sols':torch.cat(cache)
    # },
    # QPTL:{
    #     'lr':1e-4,
    #     'tau':10
    # },
    TwoStageRegression:{
        'lr':1e-3
    },
    # SPOTieBreak:{
    #     'lr':5e-4,
    #     'solver_pool':BipartiteMatchingPool()
    # }
}

for method in hparams.keys():
    for r in range(10):
        trainer = make_trainer(method.__name__)
        model = method(net=make_cora_net(), solver=BipartiteMatchingSolver(), minimize=False, **hparams[method])
        print(model.__class__.__name__)
        trainer.fit(model, train_dl, valid_dl)
        result = trainer.test(test_dataloaders=train_dl)
        d = pd.concat([pd.Series(res) for res in result])
        d['method'] = model.__class__.__name__
        res += [d]

with open(f'test_bmatching_{unique_id}_bce_on_train.csv', 'a') as f:
    pd.concat(res, axis=1).T.to_csv(f, header=f.tell()==0, index=False)