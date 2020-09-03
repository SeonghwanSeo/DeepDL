import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, args, n_char, i_to_c):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(args.n_feature, args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(args.n_feature, args.n_head, args.n_ff, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, args.n_layer)
        self.encoder = nn.Embedding(n_char, args.n_feature)
        self.n_input = args.n_feature
        self.decoder = nn.Linear(args.n_feature, n_char)
        self.src_mask = None
        self.softmax = nn.Softmax(dim=-1)
        self.start_codon = nn.Parameter(torch.zeros((args.n_feature)), requires_grad=True)
        self.n_char=n_char
        self.i_to_c=i_to_c

    def _generate_square_subsequent_mask(self, length) :
        mask = torch.triu(torch.ones(length, length)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=False):
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

    def sampling(self, max_len):
        result = ""
        with torch.no_grad() :
            codon = self.start_codon.unsqueeze(0).unsqueeze(0)
            score=0
            for _ in range(max_len) :
                device=codon.device
                mask = self._generate_square_subsequent_mask(codon.size(0)).to(device)
                self.src_mask=mask
                output = codon * math.sqrt(self.n_input)
                output = self.pos_encoder(output)
                output = self.transformer_encoder(output,self.src_mask)
                output = output[-1:][:][:]
                output = self.decoder(output)
                p_letter = self.softmax(output)
                output = torch.distributions.categorical.Categorical(p_letter)
                output = output.sample()
                letter = int(output[0][0])
                score+=-math.log(float(p_letter[0][0][letter]))
                if letter==self.n_char-1 :
                    break
                else :
                    output = self.encoder(output)
                    codon = torch.cat([codon,output],0)
                    result+=self.i_to_c[letter]
            return result, math.log(score)
        
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

