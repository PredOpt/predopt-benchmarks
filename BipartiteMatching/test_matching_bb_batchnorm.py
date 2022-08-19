import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import numpy as np
import torch
import shutil
import random
from Trainer.PO_models import DBB
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.data_utils import CoraMatchingDataModule
from Trainer.bipartite import bmatching_diverse

params_dict = {"1":{'p':0.25, 'q':0.25},"2":{'p':0.5, 'q':0.5} }


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--instance", type=str, help="{1:{'p':0.25, 'q':0.25},2:{'p':0.5, 'q':0.5}", default= "1", required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum bumber of epochs", default= 50, required=False)
parser.add_argument("--lambda_val", type=float, help="Blacbox interploation lambda paraemter", default= 20., required=False)
parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 50, required=False)
args = parser.parse_args()
###################################### Hyperparams #########################################
lambda_val= args.lambda_val
lr = args.lr
batch_size  = args.batch_size
max_epochs = args.max_epochs
seed = args.seed
params = params_dict[args.instance]
torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
###################################### Hyperparams #########################################
################## Define the outputfile
outputfile = "Rslt/DBBBatchnorm_matching{}_index{}.csv".format(args.instance, args.index)
regretfile = "Rslt/DBBBatchnorm_matchingRegret{}_index{}.csv".format(args.instance, args.index)
ckpt_dir =  "ckpt_dir/DBBBatchnorm{}_index{}/".format(args.instance, args.index)
log_dir = "lightning_logs/DBBBatchnorm{}_index{}/".format(args.instance, args.index)
learning_curve_datafile = "LearningCurve/DBBBatchnorm{}_lambdaval{}_lr{}_batchsize{}_seed{}_index{}.csv".format(args.instance,lambda_val,lr,batch_size,seed, args.index)
shutil.rmtree(log_dir,ignore_errors=True)

solver = bmatching_diverse

for seed in range(10):
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
                    monitor="val_regret",
                    dirpath=ckpt_dir, 
                    filename="model-{epoch:02d}-{val_loss:.2f}",
                    mode="min",
                )
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    data =  CoraMatchingDataModule(solver,params= params, batch_size=batch_size, generator=g, num_workers=8)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)

    trainer = pl.Trainer(max_epochs= max_epochs, min_epochs=3, logger=tb_logger, callbacks=[checkpoint_callback] )

    model = DBB(solver,lr=lr, lambda_val=lambda_val,norm=True,seed=seed)
    trainer.fit(model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    print("Model Path:",best_model_path)



    model = DBB.load_from_checkpoint(best_model_path ,solver=solver,lr=lr, lambda_val=lambda_val,norm=True,seed=seed)    

    regret_list = trainer.predict(model, data.test_dataloader())
    
    print("regrets")
    print(regret_list)
    df = pd.DataFrame({"regret":regret_list[0].tolist()})
    df.index.name='instance'
    df ['model'] = 'Blackbox'
    df['lr'] = lr
    df['lambda_val'] = lambda_val
    df['seed']= seed
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


    testresult = trainer.test(model, datamodule=data)
    df = pd.DataFrame(testresult )
    df ['model'] = 'Blackbox'
    df['lr'] = lr
    df['lambda_val'] = lambda_val
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

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
"val_mse": mses })
df['model'] = 'Blackbox'
df.to_csv(learning_curve_datafile)



