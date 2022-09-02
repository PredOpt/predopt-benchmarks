import argparse
from Trainer.data_utils import WarcraftDataModule
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
parser.add_argument("--lr", type=float, help="learning rate", default= 5e-4, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum bumber of epochs", default= 50, required=False)
parser.add_argument("--input_noise_temp", type=float, help="input_noise_temperature parameter", default= 1., required=False)
parser.add_argument("--target_noise_temp", type=float, help="target_noise_temperature parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples", default= 1, required=False)
parser.add_argument("--num_iter", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 1, required=False)

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
############### Configuration
img_size = "{}x{}".format(args.img_size, args.img_size)
###################################### Hyperparams #########################################
nb_iterations ,nb_samples= args.num_iter, args.num_samples
input_noise_temperature, target_noise_temperature = args.input_noise_temp, args.target_noise_temp
k = args.k
lr = args.lr
batch_size  = args.batch_size
max_epochs = args.max_epochs
seed = args.seed

################## Define the outputfile
outputfile = "Rslt/IMLERegret{}seed{}_index{}.csv".format(args.img_size,seed, args.index)
regretfile = "Rslt/IMLERegretRegret{}seed{}_index{}.csv".format(args.img_size,seed, args.index)
ckpt_dir =  "ckpt_dir/IMLERegret{}seed{}_index{}/".format(args.img_size,seed, args.index)
log_dir = "lightning_logs/IMLERegret{}seed{}_index{}/".format(args.img_size,seed, args.index)
learning_curve_datafile = "LearningCurve/IMLERegret{}_inptmp_{}trgttmp_{}_lr{}_batchsize{}_numsamples{}_numiter{}_seed{}_index{}.csv".format(args.img_size,input_noise_temperature, target_noise_temperature,lr,batch_size,nb_samples,nb_iterations, seed,args.index)
shutil.rmtree(log_dir,ignore_errors=True)


###################### Training Module   ######################

seed_all(seed)

g = torch.Generator()
g.manual_seed(seed)

data = WarcraftDataModule(data_dir="data/warcraft_shortest_path/{}".format(img_size), batch_size=batch_size, generator=g)
metadata = data.metadata

shutil.rmtree(ckpt_dir,ignore_errors=True)
checkpoint_callback = ModelCheckpoint(
        monitor="val_regret",
        dirpath=ckpt_dir, 
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min")

tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
trainer = pl.Trainer(max_epochs= max_epochs,  min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
model =  IMLE(metadata=metadata, nb_iterations= nb_iterations,nb_samples= nb_samples, k=k,
            input_noise_temperature= input_noise_temperature, target_noise_temperature= target_noise_temperature, lr=lr, seed=seed, loss="regret")
trainer.fit(model, datamodule=data)
best_model_path = checkpoint_callback.best_model_path
model = IMLE.load_from_checkpoint(best_model_path,
    metadata=metadata, nb_iterations= nb_iterations,nb_samples= nb_samples, 
            input_noise_temperature= input_noise_temperature, target_noise_temperature= target_noise_temperature, lr=lr, seed=seed, loss="regret")


regret_list = trainer.predict(model, data.test_dataloader())

df = pd.DataFrame({"regret":regret_list[0].tolist()})
df.index.name='instance'
df ['model'] = 'IMLE'
df['seed'] = seed
df ['batch_size'] = batch_size
df['lr'] =lr
df['k'] = k
df['input_noise_temperature'] = input_noise_temperature
df['target_noise_temperature'] = target_noise_temperature
df['nb_iterations'] = nb_iterations
df['nb_samples'] = nb_samples
with open(regretfile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)


##### SummaryWrite ######################
validresult = trainer.validate(model,datamodule=data)
testresult = trainer.test(model, datamodule=data)
df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
df ['model'] = 'IMLE'
df['seed'] = seed
df ['batch_size'] = batch_size
df['lr'] =lr
df['k'] = k
df['input_noise_temperature'] = input_noise_temperature
df['target_noise_temperature'] = target_noise_temperature
df['nb_iterations'] = nb_iterations
df['nb_samples'] = nb_samples

with open(outputfile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)

##### Save Learning Curve Data ######################
parent_dir=   log_dir+"lightning_logs/"
version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

walltimes = []
steps = []
regrets= []
mses = []
for logs in version_dirs:
    event_accumulator = EventAccumulator(logs)
    event_accumulator.Reload()

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
"val_mse": mses })
df['model'] ='IMLERegret'
df.to_csv(learning_curve_datafile,index=False)