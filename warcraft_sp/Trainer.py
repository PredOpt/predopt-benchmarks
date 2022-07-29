import imp
import pytorch_lightning as pl
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from computervisionmodels import get_model
from comb_modules.losses import HammingLoss
from comb_modules.dijkstra import ShortestPath
from metric import normalized_regret
class twostage_baseline(pl.LightningModule):
    def __init__(self, metadata, model_name= "ResNet18", arch_params={},lambda_val=20., neighbourhood_fn =  "8-grid",
     lr=1e-1,  seed=20,loss="bce"):
        """
        A class to implement two stage mse based baseline model and with test and validation module
        Args:
            model_name: ResNet for baseline
            lr: learning rate
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 
            loss: could be bce or mse
        """
        super().__init__()
        pl.seed_everything(seed)
        self.metadata = metadata
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )
        self.loss = loss
        self.lr = lr
        self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn= neighbourhood_fn)

    def forward(self,x):
        output = self.model(x)
        if self.loss=="bce":
            output = torch.sigmoid(output)

        return output

    def training_step(self, batch, batch_idx):
        input, label, true_weights = batch
        # print("input shape",input.shape,"label shape",label.shape)
        output = self(input)
        # print("Output shape", output.shape)
        
        flat_target = label.view(label.size()[0], -1)
        if self.loss == "bce":
            criterion = nn.BCELoss()
            training_loss = criterion(output, flat_target.to(torch.float32)).mean()
        if self.loss=="mse":
            criterion = nn.MSELoss(reduction='mean')
            training_loss = criterion(output, true_weights).mean()
        self.log("train_loss",training_loss ,  on_step=True, on_epoch=True, )
        return training_loss 
    def validation_step(self, batch, batch_idx):
        input, label, true_weights = batch
        output = self(input)
        # output = torch.sigmoid(output)

        if not len(output.shape) == 3:
            output = output.view(label.shape)
        relu_op = nn.ReLU()

        ######### IN the original paper, it was torch.abs() instead of Relu #########
        # weights = relu_op(output.reshape(-1, output.shape[-1], output.shape[-1]))
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = self.solver(weights)


        #flat_target = label.view(label.size()[0], -1)
 
        
        criterion1 = nn.MSELoss(reduction='mean')
        mse =  criterion1(output, true_weights).mean()
        if self.loss!= "bce":
           output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path )   

        Hammingloss = HammingLoss()(shortest_path, label)

        self.log("val_bce", bceloss, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True )
        self.log("val_mse", mse, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True )
        self.log("val_regret", regret, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True )
        self.log("val_hammingloss",  Hammingloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"val_mse":mse, "val_bce":bceloss,
             "val_regret":regret,"val_hammingloss":Hammingloss}
    # def validation_epoch_end(self, outputs):
    #     avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
    #     avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
    #     self.log("ptl/val_regret", avg_regret)
    #     self.log("ptl/val_mse", avg_mse)
    #     # self.log("ptl/val_accuracy", avg_acc)
        
    def test_step(self, batch, batch_idx):
        input, label, true_weights = batch
        output = self(input)
        # output = torch.sigmoid(output)

        if not len(output.shape) == 3:
            output = output.view(label.shape)
        relu_op = nn.ReLU()

        ######### IN the original paper, it was torch.abs() instead of Relu #########
        # weights = relu_op(output.reshape(-1, output.shape[-1], output.shape[-1]))
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = self.solver(weights)


        #flat_target = label.view(label.size()[0], -1)
 
        
        criterion1 = nn.MSELoss(reduction='mean')
        mse =  criterion1(output, true_weights).mean()
        if self.loss!= "bce":
           output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path )   

        Hammingloss = HammingLoss()(shortest_path, label)

        self.log("test_bce", bceloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test_mse", mse, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True )
        self.log("test_regret", regret, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True )
        self.log("test_hammingloss",  Hammingloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"test_mse":mse, "test_bce":bceloss,
             "test_regret":regret,"test_hammingloss":Hammingloss}

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html ###
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        # return [self.opt], [self.reduce_lr_on_plateau]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6),
                    "monitor": "val_regret",
                    # "frequency": "indicates how often the metric is updated"
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }


class Blackbox(twostage_baseline):
    def __init__(self, metadata, model_name= "CombResnet18", arch_params={},lambda_val=20., neighbourhood_fn =  "8-grid",
        lr=1e-1, seed=20,loss="hamming"):
        super().__init__(metadata, model_name, arch_params,lambda_val, neighbourhood_fn ,
        lr,  seed,loss)

        if loss=="hamming":
            self.loss_fn = HammingLoss()


    def forward(self,x):
        output = self.model(x)
        relu_op = nn.ReLU()
        return relu_op(output)

    def training_step(self, batch, batch_idx):
        input, label, true_weights = batch
        # print("input shape",input.shape,"label shape",label.shape)
        output = self(input)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = self.solver(weights)
        training_loss = self.loss_fn(shortest_path, label)

        
        # flat_target = label.view(label.size()[0], -1)
        # if self.loss == "bce":
        #     output = torch.sigmoid(output)
        #     criterion = nn.BCELoss()
        #     training_loss = criterion(output, flat_target.to(torch.float32)).mean()
        # if self.loss=="mse":
        #     criterion = nn.MSELoss(reduction='mean')
        #     training_loss = criterion(output, true_weights).mean()
        # self.log("train_loss",training_loss ,  on_step=True, on_epoch=True, )
        return training_loss 



