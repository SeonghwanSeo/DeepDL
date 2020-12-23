import numpy as np
import torch
from rdkit import Chem
import torch.nn as nn
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from utils.token import C_TO_I

class quantifier() :
    def __init__(self, model, stereo=True) : 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.c_to_i = C_TO_I
        self.stereo = stereo
        
    def __call__(self, smiles):
        device = self.device
        mol = Chem.MolFromSmiles(smiles)
        with torch.no_grad():
            if mol==None :
                print(smiles)
            if self.stereo : isomers = list(EnumerateStereoisomers(mol))
            else : isomers = [mol]
            best_score = -10000     #maximum value is the best. value = 100 + logP
            for mol in isomers :
                s = Chem.MolToSmiles(mol, isomericSmiles=True)
                s = s+'Q'
                x = torch.tensor([self.c_to_i[i] for i in list(s)]).unsqueeze(0).to(device)
                output = self.model(x)
                p_char = nn.LogSoftmax(dim = -1)(output)
                p_char = p_char.data.cpu().numpy()
                x = x.data.cpu().numpy()
                isomer_score = 100
                for i in range(len(s)):
                    isomer_score += p_char[0, i, x[0, i]]
                best_score = max(isomer_score, best_score)
        return best_score

