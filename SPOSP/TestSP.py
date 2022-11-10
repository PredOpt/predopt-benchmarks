from Trainer.PO_modelsSP import *
import pandas as pd
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint 
import random
from pytorch_lightning import loggers as pl_loggers
from Trainer.data_utils import datawrapper, ShortestPathDataModule
torch.use_deterministic_algorithms(True)
import argparse
from argparse import Namespace
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

net_layers = [nn.BatchNorm1d(5),nn.Linear(5,40)]
batchnorm_net = nn.Sequential(*net_layers)
nonorm_net = nn.Linear(5,40)
class normalized_network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.batchnorm = nn.BatchNorm1d(5)
        self.fc = nn.Linear(5,40)

    def forward(self, x):
        # x = self.batchnorm(x)
        x = self.fc(x)
        return F.normalize(x, p=1,dim = 1)
l1norm_net =  normalized_network()
network_dict = {"nonorm":nonorm_net, "batchnorm":batchnorm_net, "l1norm":l1norm_net}

parser = argparse.ArgumentParser()

parser.add_argument("--N", type=int, help="Dataset size", default= 1000, required= True)
parser.add_argument("--noise", type= float, help="noise halfwidth paraemter", default= 0.5, required= True)
parser.add_argument("--deg", type=int, help="degree of misspecifaction", default= 1000, required= True)

parser.add_argument("--model", type=str, help="name of the model", default= "", required= True)
parser.add_argument("--loss", type= str, help="loss", default= "", required=False)
parser.add_argument("--net", type=str, help="Type of Model Archietcture, one of: nonorm, batchnorm,l1norm", default= "nonorm", required= False)

parser.add_argument("--lr", type=float, help="learning rate", default= 1e-3, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs", default= 40, required=False)
parser.add_argument("--l1", type=float, help="Weight of L1 regularization", default= 1e-5, required=False)

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
network_name = argument_dict.pop('net')
net = network_dict[network_name]


N = argument_dict.pop('N')
noise = argument_dict.pop('noise')
deg = argument_dict.pop('deg')
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

################## Define the outputfile
outputfile = "Rslt/{}{}_{}{}{}_index{}.csv".format(modelname,args.loss, N,noise,deg, index)
regretfile = "Rslt/{}{}Regret_{}{}{}_index{}.csv".format(modelname,args.loss, N,noise,deg,  index)
ckpt_dir =  "ckpt_dir/{}{}_{}{}{}_index{}/".format(modelname, args.loss,N,noise,deg,   index)
log_dir = "lightning_logs/{}{}_{}{}{}_index{}/".format(modelname, args.loss, N,noise,deg,  index)
learning_curve_datafile = "LearningCurve/{}{}_{}{}{}_".format(modelname, args.loss, N,noise,deg)+".csv"

################## DataReading
Train_dfx= pd.read_csv("SyntheticData/TraindataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Train_dfy= pd.read_csv("SyntheticData/Traindatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_train =  Train_dfx.T.values.astype(np.float32)
y_train = Train_dfy.T.values.astype(np.float32)

Validation_dfx= pd.read_csv("SyntheticData/ValidationdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Validation_dfy= pd.read_csv("SyntheticData/Validationdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_valid =  Validation_dfx.T.values.astype(np.float32)
y_valid = Validation_dfy.T.values.astype(np.float32)

Test_dfx= pd.read_csv("SyntheticData/TestdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Test_dfy= pd.read_csv("SyntheticData/Testdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_test =  Test_dfx.T.values.astype(np.float32)
y_test = Test_dfy.T.values.astype(np.float32)

train_df =  datawrapper( x_train,y_train)
valid_df =  datawrapper( x_valid,y_valid)
test_df =  datawrapper( x_test,y_test)


for seed in range(10):
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    data = ShortestPathDataModule(train_df, valid_df, test_df, generator=g, num_workers=8)


    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_regret",
        dirpath=ckpt_dir, 
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
    trainer = pl.Trainer(max_epochs= argument_dict['max_epochs'],callbacks=[checkpoint_callback],  min_epochs=5, logger=tb_logger)
    if modelname=="CachingPO":
        init_cache = batch_solve(spsolver, torch.from_numpy(y_train),relaxation =False)
        model = modelcls(net= net,seed=seed, init_cache=init_cache, **argument_dict)
    else:
        model = modelcls(net= net ,seed=seed, **argument_dict)


    validresult = trainer.validate(model,datamodule=data)
    trainer.fit(model, datamodule=data)
    
    best_model_path = checkpoint_callback.best_model_path
    if modelname=="CachingPO":
        init_cache = batch_solve(spsolver, torch.from_numpy(y_train),relaxation =False)
        model = modelcls.load_from_checkpoint(best_model_path,seed=seed, net= net, init_cache=init_cache, **argument_dict)
    else:
        model = modelcls.load_from_checkpoint(best_model_path,net= net,seed=seed, **argument_dict)


    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    sol_test =  batch_solve(spsolver, torch.from_numpy(y_test).float())
    regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float(), sol_test)

    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df['seed'] =seed
    for k,v in explicit.items():
        df[k] = v
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


    ##### Summary
    validresult = trainer.validate(model,datamodule=data)
    testresult = trainer.test(model, datamodule=data)
    df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])    
    df['seed'] =seed
    for k,v in explicit.items():
        df[k] = v
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
for k,v in explicit.items():
    df[k] = v
with open( learning_curve_datafile, 'a') as f:
    df.to_csv(f, header=f.tell()==0)
