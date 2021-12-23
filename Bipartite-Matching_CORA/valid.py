import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from ray.tune.progress_reporter import CLIReporter
import torch 
print(sys.path)
try:
    from predopt_models import TwoStageRegression, Blackbox, SPO, NCECache, QPTL
except:
    print('#'*24)
    import predopt_models
    print(predopt_models, file=sys.stderr)
    print('#'*24)
from solver import BipartiteMatchingSolver
from train import get_dataloaders
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

metrics = {"mse": "ptl/val_mse", "regret": "ptl/val_regret"}
callbacks = [TuneReportCallback(metrics, on="validation_end")]

def train_tune(config, fixed_params, num_epochs=10, num_gpus=0.0, data_path='', method_cls=TwoStageRegression):
    print('#'*24)
    import predopt_models
    print(predopt_models)
    print('#'*24)
    train_dl, valid_dl, _ = get_dataloaders(data_path)
    if method_cls == NCECache:
        ## get solution cache
        cache = []
        train_dl, _, _ = get_dataloaders(data_path)
        for batch in train_dl:
            _,_, sols = batch 
            cache.append(sols)
        cache_sols = torch.stack(cache)
        model = method_cls(net=nn.Linear(5,1), solver=BipartiteMatchingSolver(), cache_sols=cache_sols, **config, **fixed_params)
    else:
        model = method_cls(net=nn.Linear(5,1), solver=BipartiteMatchingSolver(), **config, **fixed_params)
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
        'lr':tune.choice([1e-2, 1e-3]),
    },
    SPO:{
        'lr':tune.choice([1e-2,1e-3, 1e-4]),
    },
    Blackbox:{
        'lr':tune.choice([1e-2, 1e-3, 1e-4]),
        'mu':tune.choice([0.1, 1, 10, 50])
    },
    NCECache:{
        'lr':tune.choice([1e-2, 1e-3, 1e-4]),
        'variant':tune.choice(np.arange(1,5))
    },
    QPTL: {
        'lr':tune.choice([1e-2, 1e-3, 1e-4]),
        'tau': tune.choice([1e-4, 1e-2, 1, 10, 50])
    }
}

constants = defaultdict(dict) 
constants[NCECache]= {
        'psolve':1.0,
    }

def tune_asha(method=TwoStageRegression, num_samples=10, num_epochs=10, gpus_per_trial=0, data_path='data/'):
    
    config = configs[method]
    scheduler = ASHAScheduler(
        max_t=num_epochs, 
        grace_period=1,
        reduction_factor=2
    )

    fixed_params = constants[method]
    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        fixed_params=fixed_params,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        data_path=data_path,
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=['regret', 'mse', 'training_iteration']
    )

    resource_per_trial = {'cpu':1, 'gpu':gpus_per_trial}
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resource_per_trial,
        metric='regret',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        #name=f"tune_bmatch_asha_{method.__name__}"
        name='tune_bmatch_asha',
    )
    print(analysis.best_config)
    return analysis.best_result_df



if __name__ == '__main__':
    import pickle
    print(pickle.dumps(train_tune))
    for method in [TwoStageRegression]:#, SPO, Blackbox, NCECache, QPTL]:
        df:pd.DataFrame = tune_asha(method=method, num_samples=2)
        df.to_csv('toast.csv')