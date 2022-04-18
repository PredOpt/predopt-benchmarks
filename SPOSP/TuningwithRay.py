import pandas as pd
import networkx as nx
import numpy as np 
from torch import nn
import torch
import pytorch_lightning as pl
from Models import QPTL, twostage_regression, SPO,Blackbox,DCOL,IntOpt, IMLE, DPO, FenchelYoung, datawrapper
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import random
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback,  TuneReportCheckpointCallback
from ray.tune.suggest import Repeater
######################################  Data Reading #########################################
df = pd.read_csv("synthetic_path/data_N_1000_noise_0.5_deg_2.csv")
N, noise, deg = 100,0.5,1

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

################### Let's not set seed in Tuning, let it be random ###################
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(0)
# train_dl = DataLoader(train_df, batch_size= 16,worker_init_fn=seed_worker)
# valid_dl = DataLoader(valid_df, batch_size= 5,worker_init_fn=seed_worker)
# test_dl = DataLoader(test_df, batch_size= 50,worker_init_fn=seed_worker)
###############################################################################################
train_dl = DataLoader(train_df, batch_size= 32)
valid_dl = DataLoader(valid_df, batch_size= 250)
test_dl = DataLoader(test_df, batch_size= 250)

###################################### Tuning  #########################################
def model_tune(config,train_dl, valid_dl, solpool=None,num_epochs=30, num_gpus=0):
    ##### ***** Model Specific name and parameter *****
    model =  IMLE(net=nn.Linear(5,1) ,
    l1_weight = config['l1_weight'],
    k= config['k'],input_noise_temperature = config['input_noise_temperature'],
    #nb_samples = config['nb_samples'],nb_iterations = config['nb_iterations'],
    seed=random.randint(0,10))
    trainer = pl.Trainer(auto_lr_find=True)
    trainer.tune(model,train_dl, valid_dl)
    

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus= num_gpus,
        callbacks=[
            TuneReportCallback(
                {
                    "regret": "ptl/val_regret",
                    "mse": "ptl/val_mse"
                },
                on="validation_end")
        ])
    trainer.fit(model,train_dl, valid_dl)
    

def tune_model_asha(train_dl, valid_dl,solpool=None,num_samples=2, num_epochs=30, gpus_per_trial=0):
    ### ***** Model Specific config *****
    config = {
            "l1_weight":tune.grid_search([10**(k) for k in range(-5,1,2)]),
            "k":tune.grid_search([5,10]),
            "input_noise_temperature":tune.grid_search([0.1,0.5,1.0,2.0]),
            "nb_samples":tune.grid_search([1,10,50]),
            "nb_iterations":tune.grid_search([1,5,10]),
        }
    scheduler = ASHAScheduler(
         time_attr='training_iteration',
            max_t=num_epochs,
            grace_period=2,
            reduction_factor=4)

    reporter = CLIReporter(
        ### ***** Model Specific parameter *****
            parameter_columns=[ "k","input_noise_temperature", "nb_samples", "nb_iterations", "l1_weight" ],
            metric_columns=[ "training_iteration","mse", "regret"])
    analysis = tune.run(
            tune.with_parameters(
                model_tune,train_dl = train_dl, valid_dl = valid_dl,solpool=solpool,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial),
            resources_per_trial={
                "cpu": 1,
                "gpu": gpus_per_trial
            },
            metric="regret",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            ### ***** Model Specific name *****
            name="SP/")
    best_trial = analysis.get_best_trial("regret", "min", "last")
    print("Best trial final validation regret: {} mse: {}".format(
        best_trial.last_result["regret"], best_trial.last_result["mse"]))
    print("Best trial:    l1 weight {} final epoch- {}".format(   
    best_trial.config["l1_weight"],  best_trial.last_result["training_iteration"]))
    print("*************** Last Epoch ***************")
    result_df = analysis.results_df
    print( result_df.groupby(['config.k','config.input_noise_temperature',
    'config.l1_weight','config.nb_samples','config.nb_iterations']).agg({"regret":['mean','std'],
    'training_iteration':'median','time_total_s':'median'}).sort_values(by=[('regret', 'mean'), ('regret', 'std')]).to_string())
    # # print(.to_string())

    result_df = analysis.dataframe(metric="regret", mode="min")
    # print(result_df.to_string())
    print("*************** Minimal Epoch ***************")
    print(result_df.groupby(['config/k','config/input_noise_temperature',
    'config/l1_weight','config/nb_samples','config/nb_iterations']).agg({"regret":['mean','std'],
    'training_iteration':'median'}).sort_values(by=[('regret', 'mean'), ('regret', 'std')]).to_string() )
    # print(analysis.trial_dataframes.to_string() )

if __name__ == "__main__":

    tune_model_asha(train_dl,valid_dl)
