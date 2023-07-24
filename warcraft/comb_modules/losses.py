import torch
import numpy as np
import torch.nn.functional as F
class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target, true_weights):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()
        # return (torch.mean(suggested*(1.0-target)) + torch.mean((1.0-suggested)*target)) * 25.0

class RegretLoss(torch.nn.Module):
    def forward(self, suggested, target,true_weights):
        errors = true_weights * (suggested - target)
        return errors.mean(dim=0).sum()
###################################### NCE Loss  Functions  #########################################
class NCE(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1
    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm = self.mm
        for ii in range(len( pred_weights )):
            loss += ( ( mm* ( target[ii] - cache )*pred_weights[ii]  ).sum(dim=(1)) ).mean() 
        loss /= len(pred_weights)
        return loss

class NCE_c(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1
    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm = self.mm
        for ii in range(len( pred_weights )):
            loss += ( ( mm* ( target[ii] - cache )* (pred_weights[ii] - true_weights[ii])  ).sum(dim=(1)) ).mean() 
        loss /= len(pred_weights)
        return loss


class MAP(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1
    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm = self.mm

        for ii in range(len( pred_weights)):
            loss += (( mm* ( target[ii] - cache )*pred_weights[ii]  ).sum(dim=(1)) ).max() 
        loss /= len(pred_weights)
        return loss


class MAP_c(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1
    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm = self.mm

        for ii in range(len( pred_weights )):
            loss += (( mm* ( target[ii] - cache )* (pred_weights[ii] - true_weights[ii])   ).sum(dim=(1)) ).max() 
        loss /= len(pred_weights)
        return loss


class MAP_c_actual(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1
    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm = self.mm

        for ii in range(len( pred_weights)):

            loss += (( mm* ( target[ii] - cache )* (pred_weights[ii] - true_weights[ii])   ).sum(dim=(1)) ).max() 
        loss /= len(pred_weights)
        return loss

###################################### Ranking Loss  Functions  #########################################
class PointwiseLoss(torch.nn.Module):
    def forward(self, pred_weights, true_weights, target, cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        loss = 0
        for ii in range(len(pred_weights)):
            loss += ((cache*pred_weights[ii])-(cache*true_weights[ii])).square().mean()
        loss /= len(pred_weights)

        return loss

class PairwiseLoss(torch.nn.Module):
    def __init__(self, tau=0., minimize=True,mode="B"):
        super().__init__()
        self.tau = tau
        self.mm  = 1 if minimize else -1
        self.mode = mode 
    def forward(self, pred_weights, true_weights, target, cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        relu = torch.nn.ReLU()
        loss = 0
        for ii in range(len(pred_weights)):
            _,indices= np.unique((self.mm*true_weights[ii]*cache).sum(dim= (1,2)).detach().numpy(),return_index=True)

            if self.mode == 'B':
                big_ind = [indices[0] for p in range(len(indices)-1)] #good one
                small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
            if self.mode == 'W':
                big_ind = [indices[p] for p in range(len(indices)-1)] #good one
                small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
            if self.mode == 'S':
                big_ind = [indices[p] for p in range(len(indices)-1)] #good one
                small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
            
            loss += relu(  self.tau+ ( cache[big_ind]*pred_weights[ii] -cache[small_ind]*pred_weights[ii]   ).sum(dim=(1,2)) ).mean()
        loss /= len(pred_weights)
        return loss

class PairwisediffLoss(torch.nn.Module):
    def __init__(self, minimize=True,mode="B"):
        super().__init__()
        self.mm  = 1 if minimize else -1
        self.mode = mode 
    def forward(self, pred_weights, true_weights, target, cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        relu = torch.nn.ReLU()
        loss = 0
        for ii in range(len(pred_weights)):
            _,indices= np.unique((self.mm*true_weights[ii]*cache).sum(dim= (1,2)).detach().numpy(),return_index=True)

            if self.mode == 'B':
                big_ind = [indices[0] for p in range(len(indices)-1)] #good one
                small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
            if self.mode == 'W':
                big_ind = [indices[p] for p in range(len(indices)-1)] #good one
                small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
            if self.mode == 'S':
                big_ind = [indices[p] for p in range(len(indices)-1)] #good one
                small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
            
            loss += ( ( cache[big_ind]*pred_weights[ii] -cache[small_ind]*pred_weights[ii]).sum(dim=(1,2)) - ( cache[big_ind]*true_weights[ii] -cache[small_ind]*true_weights[ii]).sum(dim=(1,2)) ).square().mean()
        loss /= len(pred_weights)
        return loss


class ListwiseLoss(torch.nn.Module):
    def __init__(self, tau=1., minimize=True):
        super().__init__()
        self.tau = tau
        self.mm  = 1 if minimize else -1

    def forward(self, pred_weights, true_weights, target, cache):

        loss = 0
        mm, tau  = self.mm, self.tau

        for ii in range(len(pred_weights)):
            loss += - ( F.log_softmax((-mm*pred_weights[ii]*cache/tau).sum(dim=(1,2)),dim=0) * F.softmax((-mm*true_weights[ii]*cache/tau).sum(dim=(1,2)),dim=0)).mean()
        loss /= len(pred_weights)

        return loss