import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn
import torch
import pytorch_lightning as pl
from Models import *
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import shutil
torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



for seed in range(10):
    seed_all(seed)
    ######################################  Data Reading #########################################

    N, noise, deg = 1000,0,4
    ########## Hyperparams #########
    l1_weight,mu =  0.10000, 0.000100
    batchsize  = 32
    df = pd.read_csv("synthetic_path/data_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg))
    y = df.iloc[:,3].values
    x= df.iloc[:,4:9].values

    ######### Each instance is made of 40 edges #########
    x =  x.reshape(-1,40,5).astype(np.float32)
    y = y.reshape(-1,40).astype(np.float32)
    n_samples =  len(x)
    #######################################  Training data: 80%, validation: 5%, test: 15% #########################################
    n_training = n_samples*8//10
    n_valid = n_samples//20
    n_test = n_samples*3//20

    print("N training",n_training)
    x_train, y_train = x[:n_training], y[:n_training]
    x_valid, y_valid = x[n_training:(n_training + n_valid)], y[n_training:(n_training + n_valid)]
    x_test, y_test = x[(n_training + n_valid):], y[(n_training + n_valid):]
    print(" x test shape",y_test.shape)

    train_df =  datawrapper( x_train,y_train)
    valid_df =  datawrapper( x_valid,y_valid)
    test_df =  datawrapper( x_test,y_test)

    def seed_worker(worker_id):
        worker_seed = seed #torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(train_df, batch_size= batchsize,worker_init_fn=seed_worker,generator=g, num_workers=8)
    valid_dl = DataLoader(valid_df, batch_size= 125,worker_init_fn=seed_worker,generator=g, num_workers=8)
    test_dl = DataLoader(test_df, batch_size= 125,worker_init_fn=seed_worker,generator=g, num_workers=8)

    ######################################  Blackbox #########################################
    outputfile = "BB_rslt_autolr.csv"
    regretfile= "BB_Regret.csv"
    ckpt_dir =  "ckpt_dir/BB/"

    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
            monitor="val_regret",
            dirpath= ckpt_dir,
            filename="model-{epoch:02d}-{val_loss:.2f}",
            mode="min")
    model = Blackbox(net=nn.Linear(5,1) ,mu=mu,l1_weight=l1_weight, seed=seed)
    trainer = pl.Trainer(default_root_dir="ckpt_dir/BB/lrtuning/")
    lr_finder = trainer.tuner.lr_find(model, train_dl,valid_dl)
    # Plot with
    # fig = lr_finder.plot(suggest=True)
    # plt.show()
    try:
        suggested_lr = lr_finder.suggestion()
        suggested_lr = max(suggested_lr,1e-3)
    except TypeError:
        suggested_lr = 1e-3
    model.hparams.lr = suggested_lr


    trainer = pl.Trainer(max_epochs= 30,callbacks=[checkpoint_callback],  min_epochs=5)
    
    trainer.fit(model, train_dl,valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    model = Blackbox.load_from_checkpoint(best_model_path,
    net=nn.Linear(5,1) ,mu=mu,l1_weight=l1_weight, seed=seed)

    y_pred = model(torch.from_numpy(x_test).float()).squeeze()

    
    regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float())
    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df ['model'] = 'BB'
    df['seed'] = seed
    df ['noise'] = noise
    df ['deg'] =  deg
    df['N'] = N

    df['mu'] =mu
    df['lr'] = suggested_lr
    df['l1_weight'] = l1_weight
    # with open(regretfile, 'a') as f:
    #     df.to_csv(f, header=f.tell()==0)   

    result = trainer.test(model, dataloaders=test_dl)
    df = pd.DataFrame(result)
    df ['model'] = 'BB'
    df['seed'] = seed
    df ['noise'] = noise
    df ['deg'] =  deg
    df['N'] = N

    df['mu'] =mu
    df['lr'] = suggested_lr
    df['l1_weight'] = l1_weight
    
    # with open(outputfile, 'a') as f:
    #     df.to_csv(f, header=f.tell()==0)
