import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import types
from rdkit import Chem

from . import layers
from .default_model import DefaultModel, default_trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_utils import N_ATOM_FEATURE, get_atom_feature, get_adj
from dataset import GraphDataset

class GCNModel(DefaultModel): 
    #Static attribute
    default_parameters = {'input_size':N_ATOM_FEATURE, 'hidden_size1':32, 'hidden_size2':256, 'n_layers':4, 'n_heads':4, 'dropout':0.2}
    loss_fn = nn.BCELoss(reduction='mean')
    
    def __init__(self, params={}):
        super(GCNModel, self).__init__()
        self.initialize_hyperparams(params)
        input_size = params['input_size']
        hidden_size1 = params['hidden_size1']
        hidden_size2 = params['hidden_size2']
        n_layers = params['n_layers']
        n_heads = params['n_heads']
        dropout = params['dropout']
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU() 
        #The layers for updating graph feature
        layer_list = []
        for i in range(n_layers):
            if i==0 :
                layer_list.append(layers.GraphAttention(input_size, hidden_size1, n_heads, dropout=dropout))
            else:
                layer_list.append(layers.GraphAttention(hidden_size1, hidden_size1, n_heads, dropout=dropout))
             
        self.layer_list = nn.ModuleList(layer_list)
        
        #Readout Layer
        self.readout = nn.Linear(hidden_size1, hidden_size2)
        
        #Decode Layer
        self.decoder = nn.Linear(hidden_size2, hidden_size2)
        
        self.classifier = nn.Linear(hidden_size2, 1)
        self.trainer = gcn_trainer

    def forward(self, x, adj):
        #Update Graph feature               # [N, V, F] => [N, V, F1]
        for graphattn in self.layer_list:
            x = graphattn(x, adj)
        
        #Readout
        x = self.dropout(x)
        x = self.readout(x)                 # [N, V, F1] => [N, V, F2]
        x = self.relu(x)
        Z = torch.sum(x, 1)                 # [N, V, F2] => [N, F2]
        Z = torch.sigmoid(Z)                # Z: latent vector
        
        #Decode
        Y = self.dropout(Z)
        Y = self.decoder(Y)                 # [N, F2] => [N, F2]
        Y = self.relu(Y)
        Y = self.dropout(Y)
        Y = self.classifier(Y)              # [N, F2] => [N, 1]
        Y = torch.sigmoid(Y)                # convert logit to probability (btw 0 to 1)
        return Y

    def construct_dataset(self, data) :
        return GraphDataset(data)

    @torch.no_grad()
    def test(self, smiles: str) :
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        device = self.device
        self.eval()
        num_atoms = mol.GetNumAtoms()
        af = np.zeros((100,31))
        af[:num_atoms,:] = get_atom_feature(mol)
        adj = np.zeros((100,100))
        adj[:num_atoms,:num_atoms] = get_adj(mol)

        af = torch.from_numpy(af).to(device).float()
        adj = torch.from_numpy(adj).to(device).float()
        af, adj = af.unsqueeze(0), adj.unsqueeze(0)
        y_pred = self(af, adj)
        return float(self.forward(af, adj).item())

def gcn_trainer(model) :
    def _step(self, sample) :
        x, adj, y = sample['F'].to(self.device).float(), sample['A'].to(self.device).float(), sample['Y'].to(self.device).float()
        y_pred = self(x, adj)
        loss = GCNModel.loss_fn(y_pred.squeeze(-1), y)
        return y_pred, loss
    
    model._step = types.MethodType(_step, model)

    return default_trainer(model)
