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
from Trainer.data_utils import CoraMatchingDataModule
from Trainer.bipartite import bmatching_diverse

params_dict = {"1":{'p':0.25, 'q':0.25},"2":{'p':0.5, 'q':0.5} }


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="name of the model", default= 1e-3, required= True)
parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="inpu znd target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)


parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)


parser.add_argument("--instance", type=str, help="{1:{'p':0.25, 'q':0.25},2:{'p':0.5, 'q':0.5}", default= "1", required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum bumber of epochs", default= 50, required=False)
parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 50, required=False)


args = parser.parse_args()

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
params = params_dict[ argument_dict['instance']]
solver = bmatching_diverse(**params)

# ###################################### Hyperparams #########################################
# beta =  args.beta
# lr = args.lr
# nb_iterations ,nb_samples= args.num_iter, args.num_samples
# input_noise_temperature, target_noise_temperature = args.input_noise_temp, args.target_noise_temp
# k = args.k
# batch_size  = args.batch_size
# max_epochs = args.max_epochs
# seed = args.seed

torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
# ###################################### Hyperparams #########################################
# ################## Define the outputfile
outputfile = "Rslt/{}matching{}_index{}.csv".format(modelname,  args.instance, args.index)
regretfile = "Rslt/{}matchingRegret{}_index{}.csv".format(modelname,args.instance, args.index)
ckpt_dir =  "ckpt_dir/{}{}_index{}/".format(modelname,args.instance, args.index)
log_dir = "lightning_logs/{}{}_index{}/".format(modelname,args.instance, args.index)

learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items()] )+".csv"  

shutil.rmtree(log_dir,ignore_errors=True)
print(outputfile, regretfile, ckpt_dir, log_dir, learning_curve_datafile )


solver = bmatching_diverse(**params)

for seed in range(2):
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
                    monitor="val_regret",
                    dirpath=ckpt_dir, 
                    filename="model-{epoch:02d}-{val_regret:.8f}",
                    mode="min",
                )
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    data =  CoraMatchingDataModule(solver,params= params, 
    batch_size= argument_dict['batch_size'], generator=g, num_workers=8)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)

    model = modelcls(solver=solver,seed=seed, **argument_dict)

    trainer = pl.Trainer(max_epochs=  argument_dict['max_epochs'], min_epochs=3, 
    logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    print("Model Path:",best_model_path)



    model = modelcls.load_from_checkpoint(best_model_path ,solver=solver,seed=seed,
    **argument_dict)    

    regret_list = trainer.predict(model, data.test_dataloader())
    
    print("regrets")
    print(regret_list)
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
    print("test result")
    print(testresult)

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

