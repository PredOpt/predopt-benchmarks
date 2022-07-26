import pytorch_lightning as pl
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

def normalized_regret(true_weights, true_paths, suggested_paths, minimize=True):
    mm = 1 if minimize else -1
    suggested_paths_costs = suggested_paths * true_weights
    true_paths_costs = true_paths * true_weights
    return mm*( suggested_paths_costs - true_paths_costs).sum()/true_paths_costs.sum()