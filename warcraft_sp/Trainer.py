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
class twostage_regression(pl.LightningModule):
    def __init__(self, metadata, model_name= "ResNet18", lambda_val=20., neighbourhood_fn =  "8-grid",
     lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        """
        A class to implement two stage mse based baseline model and with test and validation module
        Args:

            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.metadata = metadata
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )
        self.lr = lr
        self.l1_weight = l1_weight
        self.max_epochs= max_epochs
        self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn= neighbourhood_fn)
        self.save_hyperparameters("lr",'l1_weight')
    def forward(self,x):
        output = self.model(x)
        output = torch.sigmoid(output)
        return output 


    def forward_pass(self, input, label, train, i):
        output = self.model(input)
        output = torch.sigmoid(output)
        flat_target = label.view(label.size()[0], -1)

        criterion = torch.nn.BCELoss()
        loss = criterion(output, flat_target).mean()
        accuracy = (output.round() * flat_target).sum() / flat_target.sum()

        suggested_path = output.view(label.shape).round()
        last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

        return loss, accuracy, last_suggestion

    def training_step(self, batch, batch_idx):
        criterion = torch.nn.BCELoss()
        
        input, label, true_weights = batch
        output = self(input)
        flat_target = label.view(label.size()[0], -1)
        training_loss = criterion(output, flat_target).mean()
        

        
        # y_hat =  self(x).squeeze()
        # criterion = nn.MSELoss(reduction='mean')
        # loss = criterion(y_hat,y)
        # l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        # training_loss =  loss  + l1penalty * self.l1_weight
        # self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        # self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",training_loss ,  on_step=True, on_epoch=True, )
        return training_loss 
    def validation_step(self, batch, batch_idx):
        input, label, true_weights = batch
        output = self(input)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])
        flat_target = label.view(label.size()[0], -1)

        criterion = torch.nn.BCELoss()
        bceloss = criterion(output, flat_target).mean()
        accuracy = (output.round() * flat_target).sum() / flat_target.sum()
        shortest_paths = self.solver(weights)
        regret = normalized_regret(true_weights, label, shortest_paths )   

        Hammingloss = HammingLoss(shortest_paths, label)

        self.log("validation_bce", bceloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("validation_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("validation_regret", regret, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("validation_hammingloss",  Hammingloss, prog_bar=True, on_step=True, on_epoch=True, )

        return {"val_accuracy":accuracy, "val_bceloss": bceloss,
             "val_regret":regret,"val_ammingloss":Hammingloss}
    # def validation_epoch_end(self, outputs):
    #     avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
    #     avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
    #     self.log("ptl/val_regret", avg_regret)
    #     self.log("ptl/val_mse", avg_mse)
    #     # self.log("ptl/val_accuracy", avg_acc)
        
    # def test_step(self, batch, batch_idx):
    #     criterion = nn.MSELoss(reduction='mean')
    #     x,y, sol = batch
    #     y_hat =  self(x).squeeze()
    #     mseloss = criterion(y_hat, y)
    #     regret_loss =  regret_fn(self.exact_solver, y_hat,y, sol) 
    #     # pointwise_loss = pointwise_crossproduct_loss(y_hat,y)

    #     self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
    #     self.log("test_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
    #     # self.log("test_pointwise", pointwise_loss, prog_bar=True, on_step=True, on_epoch=True, )

    #     return {"test_mse":mseloss, "test_regret":regret_loss}
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     num_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, epochs=self.max_epochs,
    #     steps_per_epoch = num_batches)
    #     return [optimizer], [scheduler]
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
