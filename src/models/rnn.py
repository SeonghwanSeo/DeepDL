import torch
import torch.nn as nn
import os
import types
import sys

from .default_model import DefaultModel, default_trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data_utils import N_CHAR, C_TO_I
from dataset import SmilesDataset

class RNNLM(DefaultModel) :
    #Static attribute
    default_parameters = {'input_size': N_CHAR, 'stereo': True, 'hidden_size':1024, 'n_layers':4, 'dropout':0.2}
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    def __init__(self, params = {}) :
        super(RNNLM, self).__init__()
        self.initialize_hyperparams(params)  #classmethod of DefaultModel
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        n_layers = params['n_layers']
        dropout = params['dropout']

        self.GRU = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, \
                          num_layers = n_layers, dropout = dropout)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.start_codon = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.trainer = rnnlm_trainer    #See below. Wrapper
        self.stereo = params['stereo']

    @classmethod
    def load_model(cls, model_path, map_location = 'cpu') :
        import os
        from omegaconf import OmegaConf
        parameter_path = os.path.join(model_path, 'save.pt')
        config_path = os.path.join(model_path, 'config.yaml')
        config = OmegaConf.load(config_path)
        model_params = config.model
        model = cls(model_params)
        try :
            model.initialize_model(map_location, parameter_path)
        except :
            model.start_codon = nn.Parameter(model.start_codon.view(-1))
            model.initialize_model(map_location, parameter_path)
            model.start_codon = nn.Parameter(model.start_codon.view(1,1,-1), requires_grad=True)
        return model

    def forward(self, x) :
        #================= Encoder ===================#
        self.GRU.flatten_parameters()
        x = self.embedding(x)                                   # [N, L] => [N, L, F]
        x = x.permute(1, 0, 2)                                  # [N, L, F] => [L, N, F]
        start_codon = self.start_codon.repeat(1, x.size(1), 1)  # [1, N, F]
        x = torch.cat([start_codon, x], 0)                      # [L+1, N, F]
        retval, _ = self.GRU(x)
        retval = retval.permute(1, 0, 2)                        # [L+1, N, F] => [N, L+1, F] 

        #================ Decoder ====================#
        retval = self.fc(retval)                                # [N, L+1, F] => [N, L+1, C]
                                                                # C: N_CHAR, number of class(token)
        return retval

    def construct_dataset(self, data) :
        return SmilesDataset(data, stereo=self.stereo)

    @staticmethod
    def len_mask(l, max_length) :
        """
        Mask the padding part of the result
        
        example data:
        c1ccccc1Q_________
        c1ccccc1Cc2ccccc2Q
        CNQ_______________
        ...

        We set the value of Padding part to 0
        """
        device = l.device
        mask = torch.arange(0, max_length).repeat(l.size(0)).to(device)
        l = l.unsqueeze(1).repeat(1, max_length).reshape(-1)
        mask = mask-l
        mask[mask>=0] = 0
        mask[mask<0] = 1 
        return mask

    @torch.no_grad()
    def test(self, smiles: str) :
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        if self.stereo : isomers = list(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol))
        else : isomers = [mol]

        self.eval()
        device = self.device
        best_score = -10000
        for mol in isomers :
            s = Chem.MolToSmiles(mol, isomericSmiles=True)
            s = s+'Q'
            x = torch.tensor([C_TO_I[i] for i in list(s)]).unsqueeze(0).to(device)
            output = self.forward(x)
            p_char = nn.LogSoftmax(dim = -1)(output)
            p_char = p_char.data.cpu().numpy()
            x = x.data.cpu().numpy()
            isomer_score = 0
            for i in range(len(s)):
                isomer_score += p_char[0, i, x[0, i]]
            best_score = max(isomer_score, best_score)
        return max(0, best_score + 100)


def rnnlm_trainer(model) :
    """
    ADD _step func
    """
    def _step(self, sample) :
        x, l = sample['X'].to(self.device).long(), sample['L'].to(self.device).long()
        output = self(x)[:, :-1, :]    # [N, L, C]
        mask = RNNLM.len_mask(l+1, output.size(1))
        loss = torch.sum(RNNLM.loss_fn(output.reshape(-1, N_CHAR), x.reshape(-1))*mask)/mask.sum()
        return output, loss
    
    model._step = types.MethodType(_step, model)

    return default_trainer(model)
