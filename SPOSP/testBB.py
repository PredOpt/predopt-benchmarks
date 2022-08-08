from PO_modelsSP import *
import pandas as pd
from torch.utils.data import DataLoader
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint 
import random
from pytorch_lightning import loggers as pl_loggers
from data_utils import datawrapper, ShortestPathDataModule
torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

outputfile = "Rslt/Normed_Blackbox_rslt.csv"
regretfile= "Rslt/Normed_Blackbox_Regret.csv"
ckpt_dir =  "ckpt_dir/Blackbox/"
log_dir = "lightning_logs/Blackbox/"
shutil.rmtree(log_dir,ignore_errors=True)

net_layers = [nn.BatchNorm1d(5),nn.Linear(5,40)]
normed_net = nn.Sequential(*net_layers)
############### Configuration
N, noise, deg = 100,0.0,4
for deg in [1,2,4]:
    for lr in [0.5]:
        ###################################### Hyperparams #########################################
        # lr = 0.1
        l1_weight = 1e-5
        mu = 1.
        batchsize  = 128
        max_epochs = 40
        ######################################  Data Reading #########################################

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
            trainer = pl.Trainer(max_epochs= max_epochs,callbacks=[checkpoint_callback],  min_epochs=5, logger=tb_logger)
            model = Blackbox(net= normed_net, mu=mu, lr= lr,l1_weight=l1_weight, seed=seed, max_epochs= max_epochs)
            trainer.fit(model, datamodule=data)
            best_model_path = checkpoint_callback.best_model_path
            model = Blackbox.load_from_checkpoint(best_model_path,
            net= normed_net, mu=mu, lr= lr,l1_weight=l1_weight, seed=seed, max_epochs= max_epochs)

            y_pred = model(torch.from_numpy(x_test).float()).squeeze()
            sol_test =  batch_solve(spsolver, torch.from_numpy(y_test).float())
            regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float(), sol_test)
            df = pd.DataFrame({"regret":regret_list})
            df.index.name='instance'
            df ['model'] = 'Blackbox(batchnorm)'
            df['seed'] = seed
            df ['batchsize'] = batchsize
            df ['noise'] = noise
            df ['deg'] =  deg
            df['N'] = N

            df['l1_weight'] = l1_weight
            df['lr'] = lr
            df['mu'] = mu
            with open(regretfile, 'a') as f:
                df.to_csv(f, header=f.tell()==0)


            ##### Summary
            validresult = trainer.validate(model,datamodule=data)
            testresult = trainer.test(model, datamodule=data)
            df = pd.DataFrame({**testresult[0], **validresult[0]},index=[0])
            df ['model'] = 'Blackbox(batchnorm)'
            df['seed'] = seed
            df ['batchsize'] = batchsize
            df ['noise'] = noise
            df ['deg'] =  deg
            df['N'] = N
            df['l1_weight'] = l1_weight
            df['lr'] = lr
            df['mu'] = mu
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

            events = event_accumulator.Scalars("val_regret_epoch")
            walltimes.extend( [x.wall_time for x in events])
            steps.extend([x.step for x in events])
            regrets.extend([x.value for x in events])
            events = event_accumulator.Scalars("val_mse_epoch")
            mses.extend([x.value for x in events])

        df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
        "val_mse": mses })
        df['model'] ='Blackbox'
        df.to_csv("LearningCurve/Normed_Blackbox_data_N_{}_noise_{}_deg_{}_mu{}_lr{}.csv".format(N,noise,deg,mu,lr))