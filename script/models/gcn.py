import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from .default_model import DefaultModel
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_utils import N_ATOM_FEATURE

class GCNModel(DefaultModel): 
    #Static attribute
    modeltype = 'TwoClassModel'
    require_datatype = 'Graph' 
    default_parameters = {'input_size':N_ATOM_FEATURE, 'hidden_size1':32, 'hidden_size2':256, 'n_layers':4, 'n_head':4, 'dropout':0.2}
    default_train_parameters = {'lr':1e-3, 'lr_decay':0.998}
    
    def __init__(self, params={}):
        super().__init__()
        self.initialize_params(params)
        input_size = params['input_size']
        hidden_size1 = params['hidden_size1']
        hidden_size2 = params['hidden_size2']
        n_layers = params['n_layers']
        n_head = params['n_head']
        dropout = params['dropout']
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU() 
        #The layers for updating graph feature
        layer_list = []
        for i in range(n_layers):
            if i==0 :
                layer_list.append(layers.GraphAttention(input_size, hidden_size1, n_head, dropout=dropout))
            else:
                layer_list.append(layers.GraphAttention(hidden_size1, hidden_size1, n_head, dropout=dropout))
             
        self.layer_list = nn.ModuleList(layer_list)
        
        #Readout Layer
        self.readout = nn.Linear(hidden_size1, hidden_size2)
        
        #Decode Layer
        self.decoder = nn.Linear(hidden_size2, hidden_size2)
        
        self.classifier = nn.Linear(hidden_size2, 1)

    def forward(self, x, adj):
        #Update Graph feature   [Batch, N, F] => [Batch, N, F1]
        for graphattn in self.layer_list:
            x = graphattn(x, adj)
        
        #Readout                [Batch, N, F1] => [Batch, N, F2] => [Batch, F2]
        x = self.dropout(x)
        x = self.readout(x)
        x = self.relu(x)
        Z = torch.sum(x, 1)
        Z = torch.sigmoid(Z)    #Z: latent vector, [Batch, F2]
        
        #Decode                 [Batch, F2] => [Batch, 1]
        Y = self.dropout(Z)
        Y = self.decoder(Y)
        Y = self.relu(Y)
        Y = self.classifier(Y)
        Y = torch.sigmoid(Y)
        return Y
