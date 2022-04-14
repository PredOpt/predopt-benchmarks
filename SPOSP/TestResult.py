import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn
import torch
import pytorch_lightning as pl
from Models import *
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
import random
torch.use_deterministic_algorithms(True)

def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
seed = 10 # 
for seed in range(10):
    seed_all(seed)
    ######################################  Data Reading #########################################

    N, noise, deg = 100,0.5,1
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
    train_dl = DataLoader(train_df, batch_size= 32,worker_init_fn=seed_worker,generator=g, num_workers=8)
    valid_dl = DataLoader(valid_df, batch_size= 125,worker_init_fn=seed_worker,generator=g, num_workers=8)
    test_dl = DataLoader(test_df, batch_size= 125,worker_init_fn=seed_worker,generator=g, num_workers=8)

    # #######################################  Two Stage #########################################
    outputfile = "Twostage_rslt.csv"
    regretfile= "Twostage_Regret.csv"
    ckpt_dir =  "ckpt_dir/twostage/"
    ########## Hyperparams #########
    lr, l1_weight = 0.1, 0.1
    ############ Remove Any Previous Saved models
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
                monitor="val_regret",
                dirpath= ckpt_dir,
                filename="model-{epoch:02d}-{val_loss:.2f}",
                mode="min")


    trainer = pl.Trainer(max_epochs= 3,callbacks=[checkpoint_callback],  min_epochs=5)
    model = twostage_regression(net=nn.Linear(5,1) ,lr= lr,l1_weight=l1_weight, seed=seed)
    trainer.fit(model, train_dl,valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    print("############## The selected model is:",best_model_path)
    model = twostage_regression.load_from_checkpoint(best_model_path,
    net=nn.Linear(5,1), lr= lr,l1_weight=l1_weight, seed=seed)

    y_pred = model(torch.from_numpy(x_test).float()).squeeze()

    result = trainer.test(model, dataloaders=test_dl)
    regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float())
    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df ['model'] = 'Twostage'
    df['seed'] = seed
    df ['noise'] = noise
    df ['deg'] =  deg
    df['N'] = N

    df['l1_weight'] = l1_weight
    df['lr'] = lr
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)    
    print(result)

    # df = pd.DataFrame(result)
    # df ['model'] = 'Twostage'
    # df['seed'] = seed
    # df ['noise'] = noise
    # df ['deg'] =  deg
    # df['N'] = N

    # df['l1_weight'] = l1_weight
    # df['lr'] = lr
    # with open(outputfile, 'a') as f:
    #     df.to_csv(f, header=f.tell()==0)
    ######################################  SPO #########################################
    # outputfile = "SPO_rslt.csv"
    # ckpt_dir =  "ckpt_dir/SPO/"
    # ########## Hyperparams #########
    # lr, l1_weight = 0.05, 1e-5
    # shutil.rmtree(ckpt_dir,ignore_errors=True)
    # checkpoint_callback = ModelCheckpoint(
    #         monitor="val_regret",
    #         dirpath= ckpt_dir,
    #         filename="model-{epoch:02d}-{val_loss:.2f}",
    #         mode="min")

    # trainer = pl.Trainer(max_epochs= 30,callbacks=[checkpoint_callback],  min_epochs=5)
    # model = SPO(net=nn.Linear(5,1) ,lr= lr,l1_weight=l1_weight, seed=seed)
    # trainer.fit(model, train_dl,valid_dl)
    # best_model_path = checkpoint_callback.best_model_path
    # model = SPO.load_from_checkpoint(best_model_path,
    # net=nn.Linear(5,1),lr= lr,l1_weight=l1_weight, seed=seed)

    # result = trainer.test(model, dataloaders=test_dl)
    # df = pd.DataFrame(result)
    # df ['model'] = 'SPO'
    # df['seed'] = seed
    # df ['noise'] = noise
    # df ['deg'] =  deg
    # df['N'] = N

    # df['lr'] = lr
    # df['l1_weight'] = l1_weight
    # with open(outputfile, 'a') as f:
    #     df.to_csv(f, header=f.tell()==0)

# ######################################  Blackbox #########################################
# outputfile = "Blackbox_rslt.csv"
# ckpt_dir =  "ckpt_dir/Blackbox/"
# ########## Hyperparams #########
# lr, mu = 0.01, 0.001
# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")
# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = Blackbox(net=nn.Linear(5,1) ,lr= lr,mu=mu)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = Blackbox.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1),lr= lr,mu=mu)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'Blackbox'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# df['mu'] = mu
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)
#####################################  Differentiable Convex Optimization Layers  #########################################
# outputfile = "DCOL_rslt.csv"
# ckpt_dir =  "ckpt_dir/DCOL/"
# ########## Hyperparams #########
# lr = 0.1


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = DCOL(net=nn.Linear(5,1) ,lr= lr)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = DCOL.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'DCOL'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)

#####################################  QPTL  #########################################
# outputfile = "QPTL_rslt.csv"
# ckpt_dir =  "ckpt_dir/QPTL/"
# ########## Hyperparams #########
# lr,mu = 0.1,1e-1


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = QPTL(net=nn.Linear(5,1) ,lr= lr,mu=mu)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = QPTL.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr,mu=mu)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'QPTL'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)

#####################################  IntOpt  #########################################
# outputfile = "Intopt_rslt.csv"
# ckpt_dir =  "ckpt_dir/Intopt/"
# ########## Hyperparams #########
# lr,thr,damping= 0.1,1e-1,1e-3


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = IntOpt(net=nn.Linear(5,1) ,lr= lr,thr=thr,damping=damping)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = IntOpt.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr,thr=thr,damping=damping)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'IntOpt'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)


#####################################  IMLE  #########################################
# outputfile = "IMLE_rslt.csv"
# ckpt_dir =  "ckpt_dir/IMLE/"
# ########## Hyperparams #########
# lr = 1e-3


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = IMLE(net=nn.Linear(5,1) ,lr= lr)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = IMLE.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'IMLE'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)



# #####################################  DPO  #########################################
# outputfile = "DPO_rslt.csv"
# ckpt_dir =  "ckpt_dir/DPO/"
# ########## Hyperparams #########
# lr = 1e-3


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = DPO(net=nn.Linear(5,1) ,lr= lr)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = DPO.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'DPO'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)


# #####################################  FenchelYoung  #########################################
# outputfile = "FenchelYoung_rslt.csv"
# ckpt_dir =  "ckpt_dir/FenchelYoung/"
# ########## Hyperparams #########
# lr = 1e-3


# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model = FenchelYoung(net=nn.Linear(5,1) ,lr= lr)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = FenchelYoung.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'FenchelYoung'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N

# df['lr'] = lr
# df['l1_weight'] = l1_weight
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)


#####################################  Noise Contastive Estimation  #########################################
# outputfile = "NCE_rslt.csv"
# ckpt_dir =  "ckpt_dir/NCE/"
# ########## Hyperparams #########
# lr, growth = 1e-3, 1.
# solpool = batch_solve(spsolver, torch.from_numpy(y_train),relaxation =False)

# shutil.rmtree(ckpt_dir,ignore_errors=True)
# checkpoint_callback = ModelCheckpoint(
#             monitor="val_regret",
#             dirpath= ckpt_dir,
#             filename="model-{epoch:02d}-{val_loss:.2f}",
#             mode="min")

# trainer = pl.Trainer(max_epochs= 20,callbacks=[checkpoint_callback],  min_epochs=5)
# model =  SemanticPO(loss_fn = NCE_c, solpool= solpool,net=nn.Linear(5,1), lr=lr, growth= growth)
# trainer.fit(model, train_dl,valid_dl)
# best_model_path = checkpoint_callback.best_model_path
# model = FenchelYoung.load_from_checkpoint(best_model_path,
# net=nn.Linear(5,1), lr= lr)

# result = trainer.test(model, dataloaders=test_dl)
# df = pd.DataFrame(result)
# df ['model'] = 'NCE'
# df['seed'] = seed
# df ['noise'] = noise
# df ['deg'] =  deg
# df['N'] = N
# df['growth'] = growth
# df['lr'] = lr
# with open(outputfile, 'a') as f:
#     df.to_csv(f, header=f.tell()==0)