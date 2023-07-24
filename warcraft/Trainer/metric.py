import pytorch_lightning as pl
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

def normalized_regret(true_weights, true_paths, suggested_paths, minimize=True):
    mm = 1 if minimize else -1
    suggested_paths_costs = (suggested_paths * true_weights).sum((1,2))
    true_paths_costs = (true_paths * true_weights).sum((1,2))

    return mm*(( suggested_paths_costs - true_paths_costs)/true_paths_costs).mean()


def regret_list(true_weights, true_paths, suggested_paths, minimize=True):
    mm = 1 if minimize else -1
    suggested_paths_costs = (suggested_paths * true_weights).sum((1,2))
    true_paths_costs = (true_paths * true_weights).sum((1,2))

    return mm*(( suggested_paths_costs - true_paths_costs)/true_paths_costs)

def normalized_hamming(true_weights, true_paths, suggested_paths, minimize=True):
    errors = suggested_paths * (1.0 - true_paths) + (1.0 - suggested_paths) * true_paths
    # print( errors.sum((1,2)), true_paths.sum((1,2)) )
    return (errors.sum((1,2))/true_paths.sum((1,2))).mean()


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target, true_weights):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()
        # return (torch.mean(suggested*(1.0-target)) + torch.mean((1.0-suggested)*target)) * 25.0