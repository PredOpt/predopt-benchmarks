"""
Testing Framework for Decision-Focused Learning on Bipartite Matching Problems

This script implements the experimental evaluation framework from the JAIR paper:
"Decision-focused learning: Foundations, state of the art, benchmark and future opportunities"

The framework evaluates different DFL approaches on diverse bipartite matching tasks with:
1. Various model architectures and loss functions
2. Multiple DFL methods (SPO, DBB, DPO, etc.)
3. Configurable diversity parameters (p, q)
4. Reproducible experiments through seed control

Configuration:
    Model parameters are loaded from 'config.json', enabling systematic evaluation
    of different approaches and hyperparameters.

Arguments:
    Problem Configuration:
    --instance (str): Instance type with diversity parameters, options:
                     1: {'p':0.1, 'q':0.1}
                     2: {'p':0.25, 'q':0.25}
                     3: {'p':0.5, 'q':0.5}

    Model Configuration:
    --model (str): DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')
    --loss (str): Loss function for training

    Training Parameters:
    --lr (float): Learning rate (default: 1e-3)
    --batch_size (int): Batch size (default: 128)
    --max_epochs (int): Maximum training epochs (default: 30)
    --l1_weight (float): L1 regularization weight (default: 1e-5)

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
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import numpy as np
import torch
import shutil
import random
from Trainer.PO_models import *
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.data_utils import CoraMatchingDataModule, return_trainlabel
from Trainer.bipartite import bmatching_diverse
from distutils.util import strtobool
import json

# Define diversity parameter sets for different instances
params_dict = { 
    1: {'p': 0.1, 'q': 0.1}, 
    2: {'p': 0.25, 'q': 0.25},
    3: {'p': 0.5, 'q': 0.5}  
}

parser = argparse.ArgumentParser(description="Testing framework for Decision-Focused Learning on diverse matching problems")

# Problem configuration
parser.add_argument("--model", type=str, help="Name of the DFL model to evaluate (e.g., 'SPO', 'DBB', 'DPO')", default="", required=False)
parser.add_argument("--instance", type=str, help="Instance type with diversity parameters (1, 2, or 3)", default="1", required=False)
parser.add_argument("--loss", type=str, help="Loss function for training", default="", required=False)

# Training parameters
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128, required=False)
parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=30, required=False)
parser.add_argument("--l1_weight", type=float, help="Weight of L1 regularization", default=1e-5, required=False)

# Model-specific parameters
parser.add_argument("--lambda_val", type=float, help="Interpolation parameter for blackbox differentiation", default=1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 1, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 1, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)

parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--regularizer", type=str, help="Types of Regularization", default= 'quadratic', required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--diffKKT",  action='store_true', help="Whether KKT or HSD ",  required=False)

parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 0.05)

parser.add_argument('--scheduler', dest='scheduler',  type=lambda x: bool(strtobool(x)))

class _Sentinel:
    pass
sentinel = _Sentinel()
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
# Load parameter sets from JSON file
with open('config.json', "r") as json_file:
    parameter_sets = json.load(json_file)

for parameters in parameter_sets:
    
    Args = argparse.Namespace(**parameters)
    args = parser.parse_args(namespace=Args)
    argument_dict = vars(args)
    # print(argument_dict)
    
    sentinel = _Sentinel()
    
    explicit_keys = {key: sentinel if key not in parameters else parameters[key] for key in argument_dict}
    sentinel_ns = Namespace(**explicit_keys)
    parser.parse_args(namespace=sentinel_ns)
    explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
    print ("EXPLICIT",  explicit)

    get_class = lambda x: globals()[x]
    modelcls = get_class(argument_dict['model'])
    modelname = argument_dict.pop('model')

    ######## Solver for this instance
    params = params_dict[ argument_dict['instance']]
    solver = bmatching_diverse(**params)
    if modelname=="CachingPO":
        cache = return_trainlabel( solver,params )
    # ###################################### Hyperparams #########################################

    torch.use_deterministic_algorithms(True)


    # ################## Define the outputfile
    outputfile = "Rslt/{}matching{}{}.csv".format(modelname, args.loss,  args.instance)
    regretfile = "Rslt/{}matchingRegret{}{}.csv".format(modelname,   args.loss,args.instance)
    ckpt_dir =  "ckpt_dir/{}{}{}/".format(modelname,  args.loss,args.instance)
    log_dir = "lightning_logs/{}{}{}/".format(modelname,  args.loss,args.instance)

    learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  

    shutil.rmtree(log_dir,ignore_errors=True)

    for seed in range(10):
        shutil.rmtree(ckpt_dir,ignore_errors=True)
        checkpoint_callback = ModelCheckpoint(
                        # monitor="val_regret",mode="min",
                        dirpath=ckpt_dir, 
                        filename="model-{epoch:02d}-{val_regret:.8f}",
                        
                    )
        seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        data =  CoraMatchingDataModule(solver,params= params, 
        batch_size= argument_dict['batch_size'], generator=g, num_workers=4)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
        if modelname=="CachingPO":
            model = modelcls(init_cache=cache, solver=solver,seed=seed, **argument_dict)
        else:
            model = modelcls(solver=solver,seed=seed, **argument_dict)

        trainer = pl.Trainer(max_epochs=  argument_dict['max_epochs'], min_epochs=3, 
        logger=tb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model, datamodule=data)

        best_model_path = checkpoint_callback.best_model_path
        # print("Model Path:",best_model_path)
        if modelname=="CachingPO":
            model = modelcls.load_from_checkpoint(best_model_path ,  init_cache=cache, solver=solver,seed=seed,
        **argument_dict)
        else:
            model = modelcls.load_from_checkpoint(best_model_path ,solver=solver,seed=seed,
        **argument_dict)    

        regret_list = trainer.predict(model, data.test_dataloader())
        

        df = pd.DataFrame({"regret":regret_list[0].tolist()})
        df.index.name='instance'
        for k,v in explicit.items():
            df[k] = v
        df['seed']= seed
    
        with open(regretfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)


        testresult = trainer.test(model, datamodule=data)
        df = pd.DataFrame(testresult )
        for k,v in explicit.items():
            df[k] = v
        df['seed']= seed

        with open(outputfile, 'a') as f:
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
