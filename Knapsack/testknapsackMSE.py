import pytorch_lightning as pl
import shutil
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.PO_models import twostage_mse
from Trainer.data_utils import KnapsackDataModule
from pytorch_lightning import loggers as pl_loggers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--capacity", type=int, help="capacity of knapsack", default= 12)
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
###################################### Hyperparams #########################################
lr = args.lr
batch_size  = args.batch_size
max_epochs = args.max_epochs
seed = args.seed
capacity =  args.capacity
################## Define the outputfile
outputfile = "Rslt/TwostageMSE_index{}.csv".format( args.index)
regretfile = "Rslt/TwostageMSE_Regretindex{}.csv".format( args.index)
ckpt_dir =  "ckpt_dir/TwostageMSE_index{}/".format( args.index)
log_dir = "lightning_logs/TwostageMSE_index{}/".format( args.index)
learning_curve_datafile = "LearningCurve/TwostageMSEcapa{}_lr{}_batchsize{}_index{}.csv".format(capacity,lr,batch_size, args.index)
shutil.rmtree(log_dir,ignore_errors=True)



for seed in range(10):
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)    
    data =  KnapsackDataModule(capacity=  capacity, batch_size=batch_size, generator=g)
    weights, n_items =  data.weights, data.n_items

    shutil.rmtree(ckpt_dir,ignore_errors=True)

    checkpoint_callback = ModelCheckpoint(
            monitor="val_regret",
            dirpath=ckpt_dir, 
            filename="model-{epoch:02d}-{val_loss:.2f}",
            mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
    trainer = pl.Trainer(max_epochs= max_epochs,  min_epochs=1,logger=tb_logger, callbacks=[checkpoint_callback])
    model =  twostage_mse(weights,capacity,n_items,lr=lr, seed=seed)
    trainer.fit(model, datamodule=data)
    best_model_path = checkpoint_callback.best_model_path
    model = twostage_mse.load_from_checkpoint(best_model_path,
        weights = weights,capacity= capacity,n_items = n_items,lr=lr, seed=seed)
    ##### SummaryWrite ######################
    validresult = trainer.validate(model,datamodule=data)
    testresult = trainer.test(model, datamodule=data)
    df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
    df ['model'] = 'Twostage(MSE)'
    df['seed'] = seed
    df ['batch_size'] = batch_size
    df['lr'] =lr
    df['capacity'] =capacity
    with open(outputfile, 'a') as f:
            df.to_csv(f, header=f.tell()==0)


    regret_list = trainer.predict(model, data.test_dataloader())
    

    df = pd.DataFrame({"regret":regret_list[0].tolist()})
    df.index.name='instance'
    df ['model'] = 'Twostage(MSE)'
    df['seed'] = seed
    df ['batch_size'] = batch_size
    df['lr'] =lr
    df['capacity'] =capacity
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

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
"val_mse": mses })
df['model'] = 'Twostage(MSE)'
df.to_csv(learning_curve_datafile)