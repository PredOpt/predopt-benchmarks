import imp
import pytorch_lightning as pl
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from computervisionmodels import get_model
from comb_modules.losses import HammingLoss, RegretLoss
from diff_layer import BlackboxDifflayer,SPOlayer
from comb_modules.dijkstra import get_solver, shortest_pathsolution
from metric import normalized_regret
from DPO import perturbations
from DPO import fenchel_young as fy


class twostage_baseline(pl.LightningModule):
    def __init__(self, metadata, model_name= "ResNet18", arch_params={}, neighbourhood_fn =  "8-grid",
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
        self.solver =   get_solver(neighbourhood_fn)#  ShortestPath(lambda_val=lambda_val, neighbourhood_fn= neighbourhood_fn)

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
        shortest_path =  shortest_pathsolution(self.solver, weights)  
        #flat_target = label.view(label.size()[0], -1)
 
        
        criterion1 = nn.MSELoss(reduction='mean')
        mse =  criterion1(output, true_weights).mean()
        if self.loss!= "bce":
           output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path )   

        Hammingloss = HammingLoss()(shortest_path, label, true_weights)

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
        shortest_path =  shortest_pathsolution(self.solver, weights)

        #flat_target = label.view(label.size()[0], -1)
 
        
        criterion1 = nn.MSELoss(reduction='mean')
        mse =  criterion1(output, true_weights).mean()
        if self.loss!= "bce":
           output = torch.sigmoid(output)
        criterion2 = nn.BCELoss()
        bceloss = criterion2(output, label.to(torch.float32)).mean()

        regret = normalized_regret(true_weights, label, shortest_path )   

        Hammingloss = HammingLoss()(shortest_path, label, true_weights)

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
        super().__init__(metadata, model_name, arch_params, neighbourhood_fn ,
        lr,  seed,loss)
        self.comb_layer =  BlackboxDifflayer(lambda_val=lambda_val, neighbourhood_fn= neighbourhood_fn)

        if loss=="hamming":
            self.loss_fn = HammingLoss()
        if loss=="regret":
            self.loss_fn = RegretLoss()

    def forward(self,x):
        output = self.model(x)
        relu_op = nn.ReLU()
        return relu_op(output)

    def training_step(self, batch, batch_idx):
        input, label, true_weights = batch
        # print("input shape",input.shape,"label shape",label.shape)
        output = self(input)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        shortest_path = self.comb_layer(weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        self.log("train_loss",training_loss ,  on_step=True, on_epoch=True, )
        return training_loss 


class SPO(twostage_baseline):
    def __init__(self, metadata, model_name= "CombResnet18", arch_params={}, neighbourhood_fn =  "8-grid",
        lr=1e-1, seed=20,loss="hamming"):
        super().__init__(metadata, model_name, arch_params, neighbourhood_fn ,
        lr,  seed,loss)
        self.comb_layer =  SPOlayer(neighbourhood_fn= neighbourhood_fn)
        ########### For SPO, irrespective of the final loss, the gradient would be same  #################################
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
        ### For SPO, we need the true weights as we have to compute 2*\hat{c} - c
        shortest_path = self.comb_layer(weights, label, true_weights)
        training_loss = self.loss_fn(shortest_path, label)
        self.log("train_loss",training_loss,  on_step=True, on_epoch=True, )
        return training_loss 



class FenchelYoung(twostage_baseline):
    def __init__(self, metadata, model_name= "CombResnet18", arch_params={}, neighbourhood_fn =  "8-grid",
        lr=1e-1, seed=20,loss="hamming",sigma=0.1,num_samples=10 ):
        super().__init__(metadata, model_name, arch_params, neighbourhood_fn ,
        lr,  seed,loss)
        self.sigma = sigma
        self.num_samples = num_samples
        solver =   get_solver(neighbourhood_fn)
        self.fy_solver = lambda weights: shortest_pathsolution(solver, weights)
        ########### Like SPO, The gradient does not depend on the loss function  #################################
        if loss=="hamming":
            self.loss_fn = HammingLoss()
        # if loss=="regret":
        #     self.loss_fn = RegretLoss()

    def forward(self,x):
        output = self.model(x)
        relu_op = nn.ReLU()
        return relu_op(output)

    def training_step(self, batch, batch_idx):
        criterion = fy.FenchelYoungLoss(self.fy_solver, num_samples= self.num_samples, sigma= self.sigma,maximize = False, batched= True)

        input, label, true_weights = batch
        output = self(input)
        
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        # shortest_path = self.comb_layer(weights, label, true_weights)
        
        training_loss =  criterion(weights,label).mean()
        self.log("train_loss",training_loss,  on_step=True, on_epoch=True, )
        return training_loss      

class DPO(twostage_baseline):
    def __init__(self, metadata, model_name= "CombResnet18", arch_params={}, neighbourhood_fn =  "8-grid",
        lr=1e-1, seed=20,loss="hamming",sigma=0.1,num_samples=10 ):
        super().__init__(metadata, model_name, arch_params, neighbourhood_fn ,
        lr,  seed,loss)
        self.sigma = sigma
        self.num_samples = num_samples
        solver =   get_solver(neighbourhood_fn)

        # @perturbations.perturbed(num_samples=num_samples, sigma=sigma, noise='gumbel',batched = False)
        self.dpo_solver = perturbations.perturbed(lambda weights: shortest_pathsolution(solver, -weights),
        num_samples=num_samples, sigma=sigma, noise='gumbel',batched = True)

        if loss=="hamming":
            self.loss_fn = HammingLoss()
        if loss=="regret":
            self.loss_fn = RegretLoss()

    def forward(self,x):
        output = self.model(x)
        relu_op = nn.ReLU()
        return relu_op(output)

    def training_step(self, batch, batch_idx):
        input, label, true_weights = batch
        output = self(input)
        
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        # shortest_path = self.comb_layer(weights, label, true_weights)
        
        # training_loss =  criterion(weights,label).mean()


        shortest_path = self.dpo_solver(-weights)
        training_loss = self.loss_fn(shortest_path, label, true_weights)
        self.log("train_loss",training_loss,  on_step=True, on_epoch=True, )
        return training_loss  



