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
from pytorch_lightning.callbacks import ModelCheckpoint 

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, help="size of image in one dimension", default= 12)
parser.add_argument("--mu", type=float, help="mu Paraemeter", default= 1e-4, required=False)
parser.add_argument("--lr", type=float, help="learning rate", default= 5e-4, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum bumber of epochs", default= 50, required=False)
parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 50, required=False)
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
lr = args.lr
mu = args.mu
batch_size  = args.batch_size
max_epochs = args.max_epochs
seed = args.seed

################## Define the outputfile
outputfile = "Rslt/QPTLRegret{}_index{}.csv".format(args.img_size, args.index)
ckpt_dir =  "ckpt_dir/QPTLRegret{}_index{}/".format(args.img_size, args.index)
log_dir = "lightning_logsQPTLRegret{}_index{}/".format(args.img_size, args.index)
learning_curve_datafile = "LearningCurve/QPTLRegret{}_lr{}_mu{}_batchsize{}_seed{}_index{}.csv".format(args.img_size, lr, mu, batch_size,seed, args.index)
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
# trainer = pl.Trainer( accelerator="gpu",  strategy="ddp",
# max_epochs= max_epochs,  min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
model =  QPTL(metadata=metadata, lr=lr, seed=seed,mu=mu, loss="regret")
trainer.fit(model, datamodule=data)
best_model_path = checkpoint_callback.best_model_path
model = QPTL.load_from_checkpoint(best_model_path,
    metadata=metadata, lr=lr, seed=seed,mu=mu, loss= "regret")




##### SummaryWrite ######################
validresult = trainer.validate(model,datamodule=data)
testresult = trainer.test(model, datamodule=data)
df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
df ['model'] = 'QPTL'
df['seed'] = seed
df ['batch_size'] = batch_size
df ['mu'] = mu
df['lr'] =lr
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
df['model'] ='QPTL'
df.to_csv(learning_curve_datafile,index=False)