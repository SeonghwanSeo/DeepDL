import torch
import torch.nn as nn
import math
from .default_model import DefaultModel
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_utils import N_CHAR

class TransformerModel(DefaultModel):
    #Static attribute
    modeltype = 'OneClassModel'
    require_datatype = 'SMILES'
    default_parameters = {'hidden_size':1024, 'n_head':8, 'n_ff':1024, 'n_layers':4, 'dropout':0.2, 'stereo': True} 
    default_train_parameters = {'lr':1e-4, 'lr_decay':0.99} 

    def __init__(self, params={}):
        super(TransformerModel, self).__init__()
        self.initialize_params(params)
        hidden_size = params['hidden_size']
        n_head = params['n_head']
        n_ff = params['n_ff']
        n_layers = params['n_layers']
        dropout = params['dropout']

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, n_head, n_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(N_CHAR, hidden_size)
        self.n_input = hidden_size
        self.decoder = nn.Linear(hidden_size, N_CHAR)
        self.src_mask = None
        self.start_codon = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)
        
    def forward(self, src, has_mask=True):
        src = src.permute(1, 0)                                                     #src : len*batch
        device = src.device
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.size(0)+1:
                mask = self._generate_square_subsequent_mask(src.size(0)+1)
                self.src_mask = mask.to(device)
        else:
            self.src_mask = None
        src = self.encoder(src)                                                     #len*batch => len*batch*n_feature
        startcodon=self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, src.size(1), 1)
        src=torch.cat([startcodon,src],0)                                           #len+1*batch*n_feature
        src=src*math.sqrt(self.n_input)
        src = self.pos_encoder(src)                                                 #len+1*batch*n_feature
        output = self.transformer_encoder(src, self.src_mask)            #src: len+1*batch*n_feature, mask: len+1*len+1
        output = self.decoder(output)
        output=output.permute(1,0,2)                                                #len+1* batch*n_char => batch*len+1*char
        return output
    
    @staticmethod
    def _generate_square_subsequent_mask(length) :
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=350):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

