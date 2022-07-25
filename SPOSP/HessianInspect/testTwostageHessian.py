from POHessian import *
import pandas as pd
from torch.utils.data import DataLoader
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint 
import random
from pytorch_lightning import loggers as pl_loggers
from torchviz import make_dot, make_dot_from_trace
# from pyhessian import hessian
# from density_plot import get_esd_plot
torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
outputfile = "Rslt/Normed_Twostage_rslt.csv"
regretfile= "Rslt/Normed_Twostage_Regret.csv"
ckpt_dir =  "ckpt_dir/Twostage/"
log_dir = "lightning_logs/Twostage/"
shutil.rmtree(log_dir,ignore_errors=True)

net_layers = [nn.BatchNorm1d(5),nn.Linear(5,40)]
normed_net = nn.Sequential(*net_layers)
net = nn.Linear(5,40)
############### Configuration
N, noise, deg = 100,0.5,4

###################################### Hyperparams #########################################
lr = 0.1
l1_weight = 1e-5
batchsize  = 128
max_epochs = 30

######################################  Data Reading #########################################

Train_dfx= pd.read_csv("../SyntheticData/TraindataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Train_dfy= pd.read_csv("../SyntheticData/Traindatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_train =  Train_dfx.T.values.astype(np.float32)
y_train = Train_dfy.T.values.astype(np.float32)

Validation_dfx= pd.read_csv("../SyntheticData/ValidationdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Validation_dfy= pd.read_csv("../SyntheticData/Validationdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_valid =  Validation_dfx.T.values.astype(np.float32)
y_valid = Validation_dfy.T.values.astype(np.float32)

Test_dfx= pd.read_csv("../SyntheticData/TestdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Test_dfy= pd.read_csv("../SyntheticData/Testdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_test =  Test_dfx.T.values.astype(np.float32)
y_test = Test_dfy.T.values.astype(np.float32)
# sol_test =  batch_solve(spsolver, torch.from_numpy(y_test).float())
# print("sol test shape",sol_test.shape, y_test.shape)
train_df =  datawrapper( x_train,y_train)
valid_df =  datawrapper( x_valid,y_valid)
test_df =  datawrapper( x_test,y_test)


valid_dataloader = DataLoader(valid_df, batch_size=batchsize)
hessian_dataloader = []
for i, (inputs, labels,sols) in enumerate(valid_dataloader):
    hessian_dataloader.append((inputs, labels))
    # if i == batch_num - 1:
    #     break

for seed in range(5,7):
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
    trainer = pl.Trainer(max_epochs= max_epochs,callbacks=[checkpoint_callback],  min_epochs=5, logger=tb_logger)
    model = twostage_regression(net= normed_net, lr= lr,l1_weight=l1_weight, seed=seed, max_epochs= max_epochs)
    criterion = nn.MSELoss()
        
    
    
    trainer.fit(model, datamodule=data)
    best_model_path = checkpoint_callback.best_model_path
    model = twostage_regression.load_from_checkpoint(best_model_path,
    net= normed_net, lr= lr, seed=seed)
    # dot = make_dot(model.net(torch.from_numpy(x_valid).float()), params=dict(model.net.named_parameters()))
    # print(dot)
    # dot.format = 'png'
    # dot.render("net")
    
    
    
    hessian_mat = hessian(model,criterion, x_valid,y_valid,
    figname='figures/TwostageNormed_N_{}_noise_{}_deg{}_seed{}'.format(N,noise,deg,seed))
    # eigen_plot(hessian_mat,figname='Normed_N_{}_noise_{}_deg{}_seed{}'.format(N,noise,deg,seed))
    
    
    # hessian_comp = hessian(model, criterion, dataloader= hessian_dataloader,cuda=False )
    
    # top_eigenvalues, _ = hessian_comp.eigenvalues()
    # trace = hessian_comp.trace()
    # density_eigen, density_weight = hessian_comp.density()

    # print('\n***Top Eigenvalues: ', top_eigenvalues)
    # print('\n***Trace: ', np.mean(trace))

    # # get_esd_plot(density_eigen, density_weight)    


    # y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    # # pred_df = pd.DataFrame(y_pred.detach().numpy())
    # sol_test =  batch_solve(spsolver, torch.from_numpy(y_test).float())
    # regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float(), sol_test)
    # df = pd.DataFrame({"regret":regret_list})
    # df.index.name='instance'
    # df ['model'] = 'Twostage'
    # df['seed'] = seed
    # df ['batchsize'] = batchsize
    # df ['noise'] = noise
    # df ['deg'] =  deg
    # df['N'] = N

    # df['l1_weight'] = l1_weight
    # df['lr'] = lr
    # # with open(regretfile, 'a') as f:
    # #     df.to_csv(f, header=f.tell()==0)


    # ##### Summary
    
    validresult = trainer.validate(model,datamodule=data)
    # testresult = trainer.test(model, datamodule=data)
    # df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
    # df ['model'] = 'Twostage'
    # df['seed'] = seed
    # df ['batchsize'] = batchsize
    # df ['noise'] = noise
    # df ['deg'] =  deg
    # df['N'] = N
    # df['l1_weight'] = l1_weight
    # df['lr'] = lr
    # with open(outputfile, 'a') as f:
    #     df.to_csv(f, header=f.tell()==0)
