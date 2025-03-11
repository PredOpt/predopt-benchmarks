"""
Testing Framework for Decision-Focused Learning on Energy Optimization Problems

This script implements the experimental evaluation framework from the JAIR paper:
"Decision-focused learning: Foundations, state of the art, benchmark and future opportunities"

The framework evaluates different DFL approaches on energy optimization tasks with:
1. Various model architectures and loss functions
2. Multiple DFL methods (SPO, DBB, DPO, etc.)
3. Different instance configurations
4. Reproducible experiments through seed control

Configuration:
    Model parameters are loaded from 'config.json', enabling systematic evaluation
    of different approaches and hyperparameters.

Arguments:
    Problem Configuration:
    --instance (int): Instance to solve (1, 2, or 3) (default: 1)

    Model Configuration:
    --model (str): DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')
    --loss (str): Loss function for training

    Training Parameters:
    --lr (float): Learning rate (default: 1e-3)
    --batch_size (int): Batch size (default: 64)
    --max_epochs (int): Maximum training epochs (default: 20)

    Model-Specific Parameters:
    --lambda_val (float): Interpolation parameter for blackbox differentiation (default: 1.0)
    --sigma (float): Noise parameter for DPO/FY methods (default: 1.0)
    --num_samples (int): Number of samples for FY (default: 1)
    --temperature (float): Temperature parameter for noise (default: 1.0)
    --nb_iterations (int): Number of iterations (default: 1)
    --k (int): Parameter k for specific methods (default: 10)
    --nb_samples (int): Number of samples parameter (default: 1)
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
from pytorch_lightning import loggers as pl_loggers
from Trainer.data_utils import EnergyDataModule
from Trainer.comb_solver import data_reading
from distutils.util import strtobool
import json 
import os

parser = argparse.ArgumentParser(description="Testing framework for Decision-Focused Learning on energy optimization problems")

# Problem configuration
parser.add_argument("--instance", type=int, help="Instance type to solve (1, 2, or 3)", default=1, required=False)
parser.add_argument("--model", type=str, help="Name of the DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')", default="", required=False)
parser.add_argument("--loss", type=str, help="Loss function for training", default="", required=False)

# Training parameters
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="Batch size", default=64, required=False)
parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=20, required=False)

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
parser.add_argument("--regularizer", type=str, help="Types of Regularization", default='quadratic', required=False)
parser.add_argument("--thr", type=float, help="Threshold parameter", default=1e-6, required=False)
parser.add_argument("--damping", type=float, help="Damping parameter", default=1e-8, required=False)
parser.add_argument("--diffKKT", action='store_true', help="Whether KKT or HSD", required=False)

parser.add_argument("--tau", type=float, help="Parameter of rankwise losses", default=1e-8, required=False)
parser.add_argument("--growth", type=float, help="Growth parameter of rankwise losses", default=0.05, required=False)

parser.add_argument('--scheduler', dest='scheduler', type=lambda x: bool(strtobool(x)), required=False)


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

    # Extract explicit arguments for logging
    explicit_keys = {key: sentinel if key not in parameters else parameters[key] for key in argument_dict}
    sentinel_ns = Namespace(**explicit_keys)
    parser.parse_args(namespace=sentinel_ns)
    explicit = {key: value for key, value in vars(sentinel_ns).items() if value is not sentinel}


    load = args.instance
    argument_dict = vars(args)
    get_class = lambda x: globals()[x]
    modelcls = get_class(argument_dict['model'])
    modelname = argument_dict.pop('model')

    # Configure experiment directories and files
    log_dir = f"lightning_logs/{modelname}_{argument_dict['loss']}/"
    outputfile = f"results/{modelname}_{argument_dict['loss']}.csv"
    regretfile = f"regret/{modelname}_{argument_dict['loss']}.csv"
    ckpt_dir = "checkpoints/"
    learning_curve_datafile = f"learning_curves/{modelname}_{argument_dict['loss']}.csv"

    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("regret", exist_ok=True)
    os.makedirs("learning_curves", exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Configure model checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-{val_regret:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_regret',
        mode='min'
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    # Load energy data for the specified instance
    instance = argument_dict.pop('instance')
    data = data_reading(instance)
    data = EnergyDataModule(data, batch_size=argument_dict['batch_size'])
    data.setup()
    param = data_reading("SchedulingInstances/load{}/day01.txt".format(load))

    # Clean existing logs for fresh run
    shutil.rmtree(log_dir, ignore_errors=True)

    # Repeat experiments over 10 random seeds for statistical validation
    for seed in range(10):
        seed_all(seed)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        checkpoint_callback = ModelCheckpoint(
                #monitor="val_regret",mode="min",
                dirpath=ckpt_dir, 
                filename="model-{epoch:02d}-{val_regret:.8f}",
                )

        g = torch.Generator()
        g.manual_seed(seed)    
        data =  EnergyDataModule(param =  data, batch_size= argument_dict['batch_size'], generator=g, seed= seed)

        if modelname=="CachingPO":
            cache = torch.from_numpy (data.train_df.sol)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
        trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'], 
        min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
        if modelname=="CachingPO":
            model =  modelcls(param =  data,  init_cache=cache,seed=seed, **argument_dict)
        else:
            model =  modelcls(param =  data,   seed=seed, **argument_dict)

        # Model Training and Evaluation
        # Initialize model with appropriate parameters
        trainer = pl.Trainer(
            max_epochs=argument_dict['max_epochs'],
            min_epochs=1,
            logger=tb_logger,
            callbacks=[checkpoint_callback]
        )

        # Handle special case for CachingPO model
        if modelname == "CachingPO":
            model = modelcls(param=data, init_cache=cache, seed=seed, **argument_dict)
        else:
            model =  modelcls(param =  param,   seed=seed, **argument_dict)
        validresult = trainer.validate(model,datamodule=data)
        
        # Train the model
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