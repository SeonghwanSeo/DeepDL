import torch
import torch.nn as nn
import math

class RNN(nn.Module) :
    def __init__(self, params, n_char, i_to_c) :
        super(RNN, self).__init__()
        hidden_size = params['hidden_size']
        n_layers = params['n_layers']
        dropout = params['dropout']
        self.GRU = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, num_layers = n_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_size, n_char)
        self.softmax = nn.Softmax(dim=2)
        self.embedding = nn.Embedding(n_char, hidden_size)
        self.hidden_size = hidden_size
        self.n_layer = n_layers
        self.i_to_c = i_to_c
        self.n_char = n_char
        self.start_codon = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
    
    def forward(self, x) :
        self.GRU.flatten_parameters()
        x_emb = self.embedding(x)                       #batch*len => batch*len*n_feature
        x_emb = x_emb.permute(1, 0, 2)                  #batch*len*n_feature => len*batch*n_feature
        start_codon = self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, x_emb.size(1), 1)
        input_data = torch.cat([start_codon, x_emb], 0)
        output, hidden = self.GRU(input_data)
        output = self.fc(output)                        #len*batch*n_feature => len*batch*n_char
        output = output.permute(1, 0, 2)                #len*batch*n_char => batch*len*n_char
        return output

    def sampling(self, max_len) :
        result = ""
        with torch.no_grad() :
            codon = self.start_codon.unsqueeze(0).unsqueeze(0)
            hidden = torch.zeros(self.n_layer, 1, self.hidden_size).to(codon.device)
            score=0
            for _ in range(max_len) :
                codon, hidden = self.GRU(codon, hidden)
                codon = self.fc(codon)
                p_letter = self.softmax(codon)
                codon = torch.distributions.categorical.Categorical(p_letter)
                codon = codon.sample()
                letter = int(codon[0][0])
                score+=-math.log(float(p_letter[0][0][letter]))
                if letter==self.n_char-1 :
                    break
                else :
                    codon = self.embedding(codon)
                    result+=self.i_to_c[letter]
            return result, math.log(score)
