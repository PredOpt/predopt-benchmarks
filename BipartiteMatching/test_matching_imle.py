import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import numpy as np
import torch
import shutil
import random
from Trainer.PO_models import IMLE
from pytorch_lightning.callbacks import ModelCheckpoint
from Trainer.data_utils import CoraMatchingDataModule
from Trainer.bipartite import bmatching_diverse

params_dict = {"1":{'p':0.25, 'q':0.25},"2":{'p':0.5, 'q':0.5} }


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--beta", type=float, help="parameter lambda", default= 10., required=False)
parser.add_argument("--input_noise_temp", type=float, help="input_noise_temperature parameter", default= 1., required=False)
parser.add_argument("--target_noise_temp", type=float, help="target_noise_temperature parameter", default= 1., required=False)
parser.add_argument("--num_samples", type=int, help="number of samples", default= 1, required=False)
parser.add_argument("--num_iter", type=int, help="number of iterations", default= 10, required=False)
parser.add_argument("--k", type=int, help="parameter k", default= 10, required=False)
parser.add_argument("--instance", type=str, help="{1:{'p':0.25, 'q':0.25},2:{'p':0.5, 'q':0.5}", default= "1", required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--seed", type=int, help="seed", default= 9, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum bumber of epochs", default= 50, required=False)
parser.add_argument("--output_tag", type=str, help="tag", default= 50, required=False)
parser.add_argument("--index", type=int, help="index", default= 50, required=False)
args = parser.parse_args()
###################################### Hyperparams #########################################
beta =  args.beta
lr = args.lr
nb_iterations ,nb_samples= args.num_iter, args.num_samples
input_noise_temperature, target_noise_temperature = args.input_noise_temp, args.target_noise_temp
k = args.k
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
outputfile = "Rslt/IMLE_matching{}_index{}.csv".format(args.instance, args.index)
regretfile = "Rslt/IMLE_matchingRegret{}_index{}.csv".format(args.instance, args.index)
ckpt_dir =  "ckpt_dir/IMLE{}_index{}/".format(args.instance, args.index)
log_dir = "lightning_logs/IMLE{}_index{}/".format(args.instance, args.index)
learning_curve_datafile = "LearningCurve/IMLE{}_temp{}_beta{}_lr{}_k{}_niter{}_batchsize{}_seed{}_index{}.csv".format(args.instance,input_noise_temperature, beta ,
lr,k, nb_iterations, batch_size,seed, args.index)
shutil.rmtree(log_dir,ignore_errors=True)

solver = bmatching_diverse(**params)

for seed in range(10):
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
                    monitor="val_regret",
                    dirpath=ckpt_dir, 
                    filename="model-{epoch:02d}-{val_loss:.8f}",
                    mode="min",
                )
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    data =  CoraMatchingDataModule(solver,params= params, batch_size=batch_size, generator=g, num_workers=8)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)

    trainer = pl.Trainer(max_epochs= max_epochs, min_epochs=3, logger=tb_logger, callbacks=[checkpoint_callback])



    model = IMLE(solver,k=k, nb_iterations=nb_iterations, nb_samples= nb_samples,beta=beta,
     input_noise_temperature=input_noise_temperature, target_noise_temperature=target_noise_temperature,
      lr=lr,seed=seed)
    trainer.fit(model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    print("Model Path:",best_model_path)



    model = IMLE.load_from_checkpoint(best_model_path ,solver=solver,
    k=k, nb_iterations=nb_iterations, nb_samples= nb_samples,beta=beta,
     input_noise_temperature=input_noise_temperature, target_noise_temperature=target_noise_temperature,
      lr=lr,seed=seed)    

    regret_list = trainer.predict(model, data.test_dataloader())
    
    print("regrets")
    print(regret_list)
    df = pd.DataFrame({"regret":regret_list[0].tolist()})
    df.index.name='instance'
    df ['model'] = 'IMLE'
    df['k'] = k
    df['input_noise_temperature'] = input_noise_temperature
    df['target_noise_temperature'] = target_noise_temperature
    df['nb_iterations'] = nb_iterations
    df['nb_samples'] = nb_samples
    df['beta'] = beta
    df['lr'] = lr
    df['seed']= seed
 
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


    testresult = trainer.test(model, datamodule=data)
    df = pd.DataFrame(testresult )
    df ['model'] = 'IMLE'
    df['k'] = k
    df['input_noise_temperature'] = input_noise_temperature
    df['target_noise_temperature'] = target_noise_temperature
    df['nb_iterations'] = nb_iterations
    df['nb_samples'] = nb_samples
    df['beta'] = beta
    df['lr'] = lr
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
df['model'] = 'IMLE'
df.to_csv(learning_curve_datafile)



