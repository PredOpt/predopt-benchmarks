import argparse
from argparse import Namespace
from Trainer.data_utils import WarcraftDataModule, return_trainlabel
from Trainer.Trainer import *
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import shutil
import random
from pytorch_lightning import loggers as pl_loggers
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning.callbacks import ModelCheckpoint 

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, help="size of image in one dimension", default= 12)
parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)


parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 70, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)

parser.add_argument("--lambda_val", type=float, help="interpolaton parameter blackbox", default= 1., required=False)
parser.add_argument("--sigma", type=float, help="DPO FY noise parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples FY", default= 1, required=False)

parser.add_argument("--temperature", type=float, help="input and target noise temperature parameter", default= 1., required=False)
parser.add_argument("--nb_iterations", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--nb_samples", type=int, help="Number of samples paprameter", default= 10, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda of IMLE", default= 10., required=False)


parser.add_argument("--mu", type=float, help="Regularization parameter DCOL & QPTL", default= 10., required=False)
parser.add_argument("--thr", type=float, help="threshold parameter", default= 1e-6)
parser.add_argument("--damping", type=float, help="damping parameter", default= 1e-8)
parser.add_argument("--tau", type=float, help="parameter of rankwise losses", default= 1e-8)
parser.add_argument("--growth", type=float, help="growth parameter of rankwise losses", default= 1e-8)



parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 1, required=False)

args = parser.parse_args()

class _Sentinel:
    pass
sentinel = _Sentinel()

argument_dict = vars(args)
get_class = lambda x: globals()[x]
modelcls = get_class(argument_dict['model'])
modelname = argument_dict.pop('model')
img_size = argument_dict.pop('img_size')
img_size = "{}x{}".format(img_size, img_size)
seed = argument_dict['seed']
argument_dict.pop('output_tag')
index = argument_dict.pop('index')


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
############### Configuration

# ###################################### Hyperparams #########################################
# beta =  args.beta
# nb_iterations ,nb_samples= args.num_iter, args.num_samples
# input_noise_temperature, target_noise_temperature = args.input_noise_temp, args.target_noise_temp
# k = args.k
# lr = args.lr
# batch_size  = args.batch_size
# max_epochs = args.max_epochs
# seed = args.seed

################## Define the outputfile
outputfile = "Rslt/{}{}{}seed{}_index{}.csv".format(modelname,args.loss, img_size,seed, index)
regretfile = "Rslt/{}{}{}Regretseed{}_index{}.csv".format(modelname,args.loss, img_size,seed, index)
ckpt_dir =  "ckpt_dir/{}{}{}seed{}_index{}/".format(modelname, args.loss, img_size,seed, index)
log_dir = "lightning_logs/{}{}{}seed{}_index{}/".format(modelname, args.loss, img_size,seed, index)
learning_curve_datafile = "LearningCurve/{}_".format(modelname)+"_".join( ["{}_{}".format(k,v) for k,v  in explicit.items() ]  )+".csv"
# shutil.rmtree(log_dir,ignore_errors=True)


###################### Training Module   ######################

seed_all(seed)

g = torch.Generator()
g.manual_seed(seed)

data = WarcraftDataModule(data_dir="data/warcraft_shortest_path/{}".format(img_size), 
batch_size= argument_dict['batch_size'], generator=g)
metadata = data.metadata

shutil.rmtree(ckpt_dir,ignore_errors=True)
checkpoint_callback = ModelCheckpoint(
        monitor= "val_regret",
        dirpath=ckpt_dir, 
        filename="model-{epoch:02d}-{val_loss:.8f}",
        mode="min")

tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'],
  min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
if modelname=="CachingPO":
    cache = return_trainlabel(data_dir="data/warcraft_shortest_path/{}".format(img_size))
    model = modelcls(metadata=metadata,init_cache=cache, **argument_dict)
else:
    model = modelcls(metadata=metadata,**argument_dict)

trainer.fit(model, datamodule=data)

best_model_path = checkpoint_callback.best_model_path
if modelname=="CachingPO":
    model = modelcls.load_from_checkpoint(best_model_path,metadata=metadata,init_cache=cache, **argument_dict)
else:
    model = modelcls.load_from_checkpoint(best_model_path,metadata=metadata, **argument_dict)


regret_list = trainer.predict(model, data.test_dataloader())

df = pd.DataFrame({"regret":regret_list[0].tolist()})
df.index.name='instance'
for k,v in explicit.items():
    df[k] = v
with open(regretfile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)

##### SummaryWrite ######################
validresult = trainer.validate(model,datamodule=data)
testresult = trainer.test(model, datamodule=data)
df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
for k,v in explicit.items():
    df[k] = v
with open(outputfile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)

# ##### Save Learning Curve Data ######################
# parent_dir=   log_dir+"lightning_logs/"
# version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

# walltimes = []
# steps = []
# regrets= []
# mses = []
# for logs in version_dirs:
#     event_accumulator = EventAccumulator(logs)
#     event_accumulator.Reload()

#     events = event_accumulator.Scalars("val_hammingloss_epoch")
#     walltimes.extend( [x.wall_time for x in events])
#     steps.extend([x.step for x in events])
#     regrets.extend([x.value for x in events])
#     events = event_accumulator.Scalars("val_mse_epoch")
#     mses.extend([x.value for x in events])

# df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_hammingloss": regrets,
# "val_mse": mses })
# df['model'] ='IMLEHamming'
# df.to_csv(learning_curve_datafile,index=False)