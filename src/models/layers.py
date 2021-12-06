import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module): 
    def __init__(self, in_features, out_features, n_head, dropout=0.2):
        super(GraphAttention, self).__init__()
        
        W_list = []
        attn_list = []
        self.n_head = n_head
        for i in range(n_head):
            W_list.append(nn.Linear(in_features, out_features))
            attn_list.append(nn.Parameter(torch.rand(size=(out_features, out_features))))
        
        self.W = nn.ModuleList(W_list)
        self.attn = nn.ParameterList(attn_list)

        self.A = nn.Linear(out_features*n_head, out_features, bias=False)

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.gate1 = nn.Linear(out_features, out_features, bias=False)
        self.gate2 = nn.Linear(out_features, out_features, bias=False)
        self.gatebias = nn.Parameter(torch.rand(size=(out_features,)))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, adj):
        """
        input = [, N, in_features]
        adj = [, N, N]
        W = [in_features, out_features]
        h = [, N, out_features]
        a_input = [, N, N, out_features]
        e = [, N, N]
        zero_vec == e == attention == adj
        h_prime = [, N, out_features]
        coeff = [, N, in_features]
        """
        input_tot = []
        for i in range(self.n_head):
            _h = self.dropout(h)
            _h = self.W[i](_h)
            _A = self.attn_matrix(_h, adj, self.attn[i]) 
            _h = self.relu(torch.matmul(_A, _h))
            input_tot.append(_h)

        # Concatenation
        _h = self.relu(torch.cat(input_tot, 2))
        _h = self.A(_h)

        # Fully Connected
        _h = self.dropout(_h)
        h = self.fc(h)

        num_atoms = h.size(1)
        coeff = torch.sigmoid(self.gate1(h) + self.gate2(_h) + self.gatebias.repeat(1, num_atoms).reshape(num_atoms, -1))
        retval = torch.mul(h, coeff) + torch.mul(_h, 1.-coeff)

        return retval
  
    @staticmethod
    def attn_matrix(_h, adj, attn):
        _h1 = torch.einsum('ij,ajk->aik', attn, torch.transpose(_h, 1, 2))
        _h2 = torch.matmul(_h, _h1)
        _adj = torch.mul(adj, _h2)
        _adj = torch.tanh(_adj)
        return _adj

