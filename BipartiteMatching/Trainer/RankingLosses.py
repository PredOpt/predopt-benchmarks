import torch
import numpy as np
import torch.nn.functional as F

###################################### Ranking Loss  Functions  #########################################
class PointwiseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_hat,y_true,sol_true,cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        loss = 0

        for ii in range(len( y_hat )):
            loss += ((cache*y_hat[ii])-(cache*y_true[ii])).square().mean() 
        loss /= len(y_hat)

        return loss
class ListwiseLoss(torch.nn.Module):
    def __init__(self, temperature=0., minimize=False):
        super().__init__()
        self.temperature = temperature
        self.mm  = 1 if minimize else -1
    def forward(self, y_hat,y_true,sol_true,cache):

        loss = 0
        mm, temperature  = self.mm, self.temperature

        for ii in range(len( y_hat )):
            loss += - ( F.log_softmax((-mm*y_hat[ii]*cache/temperature).sum(dim=(1)),dim=0) * F.softmax((-mm*y_true[ii]*cache/temperature).sum(dim=(1)),dim=0)).mean()
        loss /= len(y_hat)

        return loss


class PairwisediffLoss(torch.nn.Module):
    def __init__(self, minimize=True):
        super().__init__()
        self.mm  = 1 if minimize else -1

    def forward(self, y_hat,y_true,sol_true,cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        
        loss = 0
        for ii in range(len( y_hat )):
            _,indices= np.unique((self.mm*y_true[ii]*cache).sum(dim= (1)).detach().numpy(),return_index=True)

            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
        
            
            loss += ( ( cache[big_ind]*y_hat[ii] -cache[small_ind]*y_hat[ii]).sum(dim=(1)) - ( cache[big_ind]*y_true[ii] -cache[small_ind]*y_true[ii]).sum(dim=(1)) ).square().mean()
        loss /= len(y_hat)
        return loss

class PairwiseLoss(torch.nn.Module):
    def __init__(self, margin=0., minimize=True):
        super().__init__()
        self.margin = margin
        self.mm  = 1 if minimize else -1
    def forward(self, y_hat,y_true,sol_true,cache):
        '''
        pred_weights: predicted cost vector [batch_size, img,img]
        true_weights: actua cost vector [batch_size, img,img]
        target: true shortest path [batch_size, img,img]
        cache: cache is torch array [cache_size, img,img]
        '''
        relu = torch.nn.ReLU()
        loss = 0
        mm, margin  = self.mm, self.margin
        for ii in range(len( y_hat )):
            _,indices= np.unique((self.mm*y_true[ii]*cache).sum(dim= (1)).detach().numpy(),return_index=True)

            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
            
            loss += relu(  margin + mm*( cache[big_ind]*y_hat[ii] -cache[small_ind]*y_hat[ii] ).sum(dim=(1)) ).mean()
        loss /= len(y_hat)
        return loss