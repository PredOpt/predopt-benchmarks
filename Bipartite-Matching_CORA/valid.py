import sys
import os
from numpy import fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.tune.stopper import TrialPlateauStopper
import pandas as pd
from ray.tune.progress_reporter import CLIReporter
import torch 
import psutil 
import gc
from predopt_models import TwoStageRegression, Blackbox, SPO, NCECache, QPTL, SPOTieBreak
from solver import BipartiteMatchingSolver, BipartiteMatchingPool
from train import get_dataloaders, make_cora_net
import pytorch_lightning as pl
import torch.nn as nn
import ray
from ray import tune
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from ray.tune.schedulers import ASHAScheduler
import math
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from collections import defaultdict
ray.init(local_mode=True)
metrics = {"mse": "ptl/val_mse", "regret": "ptl/val_regret"}
callbacks = [TuneReportCallback(metrics, on="validation_end")]

def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

def train_tune(config, fixed_params, num_epochs=10, num_gpus=0.0, data_path='', method_cls=TwoStageRegression):
    auto_garbage_collect()
    train_dl, valid_dl, _ = get_dataloaders(data_path)
    if method_cls == NCECache:
        ## get solution cache
        cache = []
        train_dl, _, _ = get_dataloaders(data_path)
        for batch in train_dl:
            _,_, sols = batch 
            cache.append(sols)
        cache_sols = torch.cat(cache)
        model = method_cls(net=make_cora_net(), solver=BipartiteMatchingSolver(),cache_sols=cache_sols, **config, **fixed_params)
    elif method_cls == SPOTieBreak:
        model = method_cls(net=make_cora_net(), solver=BipartiteMatchingSolver(), solver_pool=BipartiteMatchingPool(), **config, **fixed_params)
    else:
        model = method_cls(net=make_cora_net(), solver=BipartiteMatchingSolver(), twostage_criterion=nn.BCELoss(), **config, **fixed_params)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),
        # avoid log duplicate
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."
        ), 
        progress_bar_refresh_rate=0,
        callbacks = callbacks
    )
    trainer.fit(model, train_dl, valid_dl)

configs = {
    TwoStageRegression: {
        'lr':tune.grid_search([1e-2, 5e-3, 1e-3,5e-4]),
    },
    # SPO:{
    #     'lr':tune.grid_search([1e-2,1e-3, 1e-4]),
    # },
    # SPOTieBreak:{
    #     'lr':tune.grid_search([1e-2, 1e-3, 5e-3, 1e-4, 5e-4])
    # },
    # Blackbox:{
    #     'lr':tune.grid_search([5e-4, 1e-3, 1e-4]),
    #     'mu':tune.grid_search([0.1, 1, 10, 50])
    # },
    # NCECache:{
    #     'lr':tune.grid_search([5e-4, 1e-3, 5e-3, 1e-4]),
    #     #'lr':tune.choice([1e-3]),
    #     'variant':tune.grid_search(list(np.arange(1,5)))
    # },
    # QPTL: {
    #     'lr':tune.grid_search([1e-2, 1e-3, 1e-4]),
    #     'tau': tune.grid_search([1e-4, 1e-2, 1, 10, 50])
    # }
}

constants = defaultdict(dict) 
constants[NCECache]= {
        'psolve':1.0,
    }

def tune_asha(method=TwoStageRegression, num_samples=5, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1, n_workers=1, data_path='../data/'):
    
    config = configs[method]
    scheduler = ASHAScheduler(
        max_t=num_epochs, 
        grace_period=1,
        reduction_factor=2
    )

    fixed_params = constants[method]
    fixed_params['minimize'] = False
    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        fixed_params=fixed_params,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        data_path=data_path,
        method_cls = method
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=['regret', 'mse', 'training_iteration']
    )

    resource_per_trial = {'cpu':cpus_per_trial, 'gpu':gpus_per_trial}
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resource_per_trial,
        metric='regret',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_concurrent_trials=n_workers,
        name=f"tune_bmatch_asha_{method.__name__}",
    )
    #print(analysis.best_config)
    return analysis.best_result_df



if __name__ == '__main__':
    import multiprocessing as mp
    list_df = []
    num_workers = min(mp.cpu_count(),6 )
    for method in [TwoStageRegression]:#[SPOTieBreak, NCECache, SPO, Blackbox, NCECache, QPTL]:
        df:pd.DataFrame = tune_asha(method=method, num_samples=5, cpus_per_trial=1, num_epochs=10, n_workers=num_workers,
         data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
        df['method'] = method.__name__
        with open('hparam_tune_matching_BCE_2S.csv', 'a') as f: 
            df.to_csv(f, index=False, header=f.tell()==0)
        
        
        