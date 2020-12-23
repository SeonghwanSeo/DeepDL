import torch
import torch.nn as nn
from .default_model import DefaultModel
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_utils import N_CHAR

class RNN(DefaultModel) :
    #Static attribute
    modeltype = 'OneClassModel'
    require_datatype = 'SMILES'
    default_parameters = {'input_size': N_CHAR, 'hidden_size':1024, 'n_layers':4, 'dropout':0.2}
    default_parameters_keys = ['hidden_size', 'n_layers', 'dropout']
    default_train_parameters = {'lr':1e-4, 'lr_decay':0.99} 

    def __init__(self, params = {}) :
        super().__init__()
        self.initialize_params(params)  #classmethod of DefaultModel
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        n_layers = params['n_layers']
        dropout = params['dropout']

        self.GRU = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, num_layers = n_layers, dropout = dropout)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.start_codon = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x) :
        #================= Encoder ===================#
        self.GRU.flatten_parameters()
        x = self.embedding(x)                       #batch*len => batch*len*hidden_size
        x = x.permute(1, 0, 2)                  #batch*len*hidden_size => len*batch*hidden_size
        start_codon = self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, x.size(1), 1)
        x = torch.cat([start_codon, x], 0)
        retval, _ = self.GRU(x)
        retval = retval.permute(1, 0, 2)                #len*batch*n_char => batch*len*n_char

        #================ Decoder ====================#
        retval = self.fc(retval)                        #len*batch*hidden_size => len*batch*n_char

        return retval

