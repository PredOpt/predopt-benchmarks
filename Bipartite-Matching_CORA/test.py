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
from solver import BipartiteMatchingSolver
from predopt_models import SPO, Blackbox, NCECache, QPTL, TwoStageRegression
from torch.utils.data import DataLoader

######################################  Data Reading #########################################
train_dl, valid_dl, test_dl = get_dataloaders(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
unique_id = str(time.time_ns())[4:]
callbacks = [EarlyStopping(monitor="val_regret", patience=2)]
def make_trainer(name):
    return  pl.Trainer(max_epochs= 20, 
    logger=TensorBoardLogger(
            save_dir=os.path.dirname(os.path.abspath(__file__)), name=f"runs/{name}_{unique_id}", version="."
        ),
    callbacks=callbacks
     )
res = []
######################################  Two Stage #########################################
trainer = make_trainer('2S-MSE')
model = TwoStageRegression(net=make_cora_net() , solver=BipartiteMatchingSolver(), lr= 1e-3)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "2S-MSE"
res += [d]

trainer = make_trainer('2S-CE')
model = TwoStageRegression(net=make_cora_net() , solver=BipartiteMatchingSolver(), lr= 1e-3, twostage_criterion=nn.BCELoss())
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "2S-CE"
res += [d]
# ######################################  SPO #########################################
trainer = make_trainer('SPO')
model = SPO(net=make_cora_net() , solver=BipartiteMatchingSolver(), lr= 1e-3)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "SPO"
res += [d]
# ######################################  Blackbox #########################################
trainer = make_trainer('BB')
model = Blackbox(net=make_cora_net() , solver=BipartiteMatchingSolver(), lr= 1e-4, mu=1)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "BB"
res += [d]
# #####################################  Contrastive Loss & Solution caching  #########################################
cache = []
for batch in train_dl:
    _,_, sols = batch 
    cache.append(sols)
cache_sols = torch.cat(cache)
trainer = make_trainer('NCE100')
model = NCECache(net=make_cora_net() , solver=BipartiteMatchingSolver(), cache_sols= cache_sols,lr= 1e-3, psolve=1, variant=4)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "NCE100"
res += [d]

trainer = make_trainer('NCE10')
model = NCECache(net=make_cora_net() , solver=BipartiteMatchingSolver(), cache_sols= cache_sols,lr= 1e-3, psolve=0.1, variant=4)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "NCE10"
res += [d]
#####################################  QPTL  #########################################
trainer = make_trainer('QPTL')
model = QPTL(net=make_cora_net() , solver=BipartiteMatchingSolver(), lr= 0.01, tau=0.01)
trainer.fit(model, train_dl, valid_dl)
result = trainer.test(test_dataloaders=test_dl)
d = pd.Series(result[0])
d['method'] = "QPTL"
res += [d]

pd.concat(res, axis=1).T.to_csv(f"test_bmatching_{unique_id}.csv")