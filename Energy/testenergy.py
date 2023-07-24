import argparse
from argparse import Namespace
import pytorch_lightning as pl
import shutil
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.PO_models import *
from pytorch_lightning import loggers as pl_loggers
from Trainer.data_utils import EnergyDataModule
from Trainer.comb_solver import data_reading
from distutils.util import strtobool
parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, help="Solve for instance 1, 2 or 3?", default= 1, required= True)
parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)

parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 64, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 20, required=False)
# parser.add_argument("--l1_weight", type=float, help="Weight of L1 regularization", default= 1e-5, required=False)


parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--regularizer", type=str, help="Types of Regularization", default= 'quadratic', required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--diffKKT",  action='store_true', help="Whether KKT or HSD ",  required=False)

parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)

parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))
args = parser.parse_args()
load = args.instance

class _Sentinel:
    pass
sentinel = _Sentinel()

argument_dict = vars(args)
get_class = lambda x: globals()[x]
modelcls = get_class(argument_dict['model'])
modelname = argument_dict.pop('model')
sentinel_ns = Namespace(**{key:sentinel for key in argument_dict})
parser.parse_args(namespace=sentinel_ns)

explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }


torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


################## Define the outputfile
outputfile = "Rslt/RuntimeRun.{}{}Energy.csv".format(modelname,args.loss)
regretfile = "Rslt/RuntimeRun.{}{}EnergyRegret.csv".format( modelname,args.loss )
ckpt_dir =  "ckpt_dir/{}{}/".format( modelname,args.loss)
log_dir = "lightning_logs/{}{}/".format(  modelname,args.loss )
learning_curve_datafile = "LearningCurve/RuntimeRun.{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  


param = data_reading("SchedulingInstances/load{}/day01.txt".format(load))

shutil.rmtree(log_dir,ignore_errors=True)

for seed in range(10):
    seed_all(seed)
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
            #monitor="val_regret",mode="min",
            dirpath=ckpt_dir, 
            filename="model-{epoch:02d}-{val_regret:.8f}",
            )

    g = torch.Generator()
    g.manual_seed(seed)    
    data =  EnergyDataModule(param =  param, batch_size= argument_dict['batch_size'], generator=g, seed= seed)

    if modelname=="CachingPO":
        cache = torch.from_numpy (data.train_df.sol)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
    trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], 
    min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
    if modelname=="CachingPO":
        model =  modelcls(param =  param,  init_cache=cache,seed=seed, **argument_dict)
    else:
        model =  modelcls(param =  param,   seed=seed, **argument_dict)
    validresult = trainer.validate(model,datamodule=data)
    
    trainer.fit(model, datamodule=data)
    best_model_path = checkpoint_callback.best_model_path

    if modelname=="CachingPO":
        model =  modelcls.load_from_checkpoint(best_model_path,
        param =  param,  init_cache=cache,seed=seed, **argument_dict)
    else:
        model =  modelcls.load_from_checkpoint(best_model_path,
        param =  param,   seed=seed, **argument_dict)
   ##### SummaryWrite ######################
    validresult = trainer.validate(model,datamodule=data)
    testresult = trainer.test(model, datamodule=data)
    df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
    for k,v in explicit.items():
        df[k] = v
    df['seed'] = seed
    df['instance'] = load
    with open(outputfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)


    regret_list = trainer.predict(model, data.test_dataloader())
    

    df = pd.DataFrame({"regret":regret_list[0].tolist()})
    df.index.name='instance'
    for k,v in explicit.items():
        df[k] = v
    df['seed'] = seed
    df['instance'] = load    
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)



###############################  Save  Learning Curve Data ########
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parent_dir=   log_dir+"lightning_logs/"
version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

walltimes = []
steps = []
regrets= []
mses = []
for logs in version_dirs:
    event_accumulator = EventAccumulator(logs)
    event_accumulator.Reload()

    events = event_accumulator.Scalars("val_regret")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
"val_mse": mses })
df['model'] = modelname
df.to_csv(learning_curve_datafile)