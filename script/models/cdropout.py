import torch
import torch.nn as nn
import numpy as np

# Copyright to https://github.com/yaringal/ConcreteDropout
# Yarin Gal, Jiri Hron, Alex Kendall, Concrete Dropout, NIPS 2017

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer = 1e-6, dropout_regularizer=1e-3, init_min = 0.1, init_max = 0.1):
        super().__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.rand(1).uniform_(init_min, init_max))
        self.rr = 0

    def forward(self, x, layer):

        p = torch.sigmoid(self.p_logit)
        retval = layer(self._concrete_dropout(x, p))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        kr = self.weight_regularizer * sum_of_squares / (1. - p) 
        dr = p * torch.log(p) + (1. - p) * torch.log(1. - p)
        
        input_dim = x[0].numel()
        dr *= self.dropout_regularizer * input_dim

        self.rr = kr + dr
        return retval
    
    def _concrete_dropout(self, x, p):
        eps = 1e-07
        temp = 0.1
        
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p + eps)
                    - torch.log(1. - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1. - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob/temp)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - p
        
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x


