"""
Testing Framework for Decision-Focused Learning on Knapsack Problems

This script implements the experimental evaluation framework from the JAIR paper:
"Decision-focused learning: Foundations, state of the art, benchmark and future opportunities"

The framework evaluates different DFL approaches on knapsack prediction tasks with:
1. Various model architectures and loss functions
2. Multiple DFL methods (SPO, DBB, DPO, etc.)
3. Configurable knapsack parameters
4. Reproducible experiments through seed control

Configuration:
    Model parameters are loaded from 'config.json', enabling systematic evaluation
    of different approaches and hyperparameters.

Arguments:
    Problem Configuration:
    --capacity (int): Capacity of knapsack (default: 12)

    Model Configuration:
    --model (str): DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')
    --loss (str): Loss function for training
    
    Model-Specific Parameters:
    --lambda_val (float): Interpolation parameter for blackbox differentiation (default: 1.0)
    --sigma (float): Noise parameter for DPO/FY methods (default: 1.0)
    --num_samples (int): Number of samples for FY (default: 1)
    --temperature (float): Temperature parameter for noise (default: 1.0)
    --nb_iterations (int): Number of iterations (default: 1)
    --k (int): Parameter k for specific methods (default: 10)
    --nb_samples (int): Number of samples parameter (default: 1)
    --beta (float): Parameter lambda of IMLE (default: 10.0)
    --mu (float): Regularization parameter DCOL & QPTL (default: 10.0)
    --thr (float): Threshold parameter (default: 1e-6)
    --damping (float): Damping parameter (default: 1e-8)
    --tau (float): Parameter of rankwise losses (default: 1e-8)
    --growth (float): Growth parameter of rankwise losses (default: 0.05)
    --lr (float): Learning rate (default: 1e-3)
    --batch_size (int): Batch size (default: 128)
    --max_epochs (int): Maximum number of epochs (default: 35)
    --num_workers (int): Maximum number of workers (default: 4)
    --scheduler (bool): Use scheduler (default: False)
"""

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
from Trainer.data_utils import KnapsackDataModule
from pytorch_lightning import loggers as pl_loggers
from distutils.util import strtobool
import json
import os

parser = argparse.ArgumentParser(description="Testing framework for Decision-Focused Learning on knapsack problems")

# Problem configuration
parser.add_argument("--capacity", type=int, help="Capacity of knapsack", default=12, required=False)

# Model configuration
parser.add_argument("--model", type=str, help="Name of the DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')", default="", required=False)
parser.add_argument("--loss", type=str, help="Loss function for training", default="", required=False)

# Model-specific parameters
parser.add_argument("--lambda_val", type=float, help="Interpolation parameter for blackbox differentiation", default=1., required=False)
parser.add_argument("--sigma", type=float, help="Noise parameter for DPO/FY methods", default=1., required=False)
parser.add_argument("--num_samples", type=int, help="Number of samples for FY", default=1, required=False)
parser.add_argument("--temperature", type=float, help="Temperature parameter for noise", default=1., required=False)
parser.add_argument("--nb_iterations", type=int, help="Number of iterations", default=1, required=False)
parser.add_argument("--k", type=int, help="Parameter k for specific methods", default=10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples parameter", default=1, required=False)
parser.add_argument("--beta", type=float, help="Parameter lambda of IMLE", default=10., required=False)
parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default=10., required=False)
parser.add_argument("--thr", type=float, help="Threshold parameter", default=1e-6, required=False)
parser.add_argument("--damping", type=float, help="Damping parameter", default=1e-8, required=False)
parser.add_argument("--tau", type=float, help="Parameter of rankwise losses", default=1e-8, required=False)
parser.add_argument("--growth", type=float, help="Growth parameter of rankwise losses", default=0.05, required=False)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128, required=False)
parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=35, required=False)
parser.add_argument("--num_workers", type=int, help="Maximum number of workers", default=4, required=False)
parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))

class _Sentinel:
    pass
sentinel = _Sentinel()

def seed_all(seed):
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed (int): Random seed value
    """
    print("[ Using Seed : ", seed, " ]")
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Load parameter sets from JSON file
with open('config.json', "r") as json_file:
    parameter_sets = json.load(json_file)

for parameters in parameter_sets:
    # Parse arguments from config file
    Args = argparse.Namespace(**parameters)
    args = parser.parse_args(namespace=Args)
    argument_dict = vars(args)
    
    sentinel = _Sentinel()
    
    explicit_keys = {key: sentinel if key not in parameters else parameters[key] for key in argument_dict}
    sentinel_ns = Namespace(**explicit_keys)
    parser.parse_args(namespace=sentinel_ns)
    explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
    # print ("EXPLICIT",  explicit)
    get_class = lambda x: globals()[x]
    modelcls = get_class(argument_dict['model'])
    modelname = argument_dict.pop('model')
    capacity= argument_dict.pop('capacity')
    torch.use_deterministic_algorithms(True)

    ################## Define the outputfile
    outputfile = "Rslt/{}Knapsack{}.csv".format(modelname,args.loss)
    regretfile = "Rslt/{}KnapsackRegret{}.csv".format( modelname,args.loss )
    ckpt_dir =  "ckpt_dir/{}{}/".format( modelname,args.loss)
    log_dir = "lightning_logs/{}{}/".format(  modelname,args.loss )
    learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  

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
        data =  KnapsackDataModule(capacity=  capacity, batch_size= argument_dict['batch_size'], generator=g, seed= seed, num_workers= args.num_workers)
        weights, n_items =  data.weights, data.n_items
        if modelname=="CachingPO":
            cache = torch.from_numpy (data.train_df.sol)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
        trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], 
        min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
        if modelname=="CachingPO":
            model =  modelcls(weights,capacity,n_items,init_cache=cache,seed=seed, **argument_dict)
        else:
            model =  modelcls(weights,capacity,n_items,seed=seed, **argument_dict)
        validresult = trainer.validate(model,datamodule=data)
        
        trainer.fit(model, datamodule=data)
        best_model_path = checkpoint_callback.best_model_path

        # Load best model checkpoint for evaluation
        if modelname=="CachingPO":
            model =  modelcls.load_from_checkpoint(best_model_path,
            weights = weights,capacity= capacity,n_items = n_items,init_cache=cache,seed=seed, **argument_dict)
        else:
            model =  modelcls.load_from_checkpoint(best_model_path,
            weights = weights,capacity= capacity,n_items = n_items,seed=seed, **argument_dict)
        
        validresult = trainer.validate(model,datamodule=data)
        testresult = trainer.test(model, datamodule=data)
        df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
        for k,v in explicit.items():
            df[k] = v
        df['seed'] = seed
        df['capacity'] = capacity
        with open(outputfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)

        # Calculate and save regret values
        regret_list = trainer.predict(model, data.test_dataloader())
        

        df = pd.DataFrame({"regret":regret_list[0].tolist()})
        df.index.name='instance'
        for k,v in explicit.items():
            df[k] = v
        df['seed'] = seed
        df['capacity'] =capacity    
        with open(regretfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)

        # Save learning curves from TensorBoard logs
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



        
