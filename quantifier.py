import numpy as np
import torch
from rdkit import Chem
import torch.nn as nn
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

class quantifier() :
    def __init__(self, model, c_to_i, stereo=True, reduction = 'sum') : 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.c_to_i = c_to_i
        self.stereo = stereo
        self.reduction = reduction
        self.nondrug = 'c1ccccc1'      #nondrug name(To be modified)
        self.nondrug_score = self.mol_to_score(self.nondrug)
        self.drug='C1CCCCC1'           #drug name(To be modified)
        self.drug_score = self.mol_to_score(self.drug)
        
    def __call__(self, smiles) :
        _, score = self.mol_to_score(smiles)
        return self.normalize(score)

    def mol_to_score(self, smiles):
        device = self.device
        mol = Chem.MolFromSmiles(smiles)
        if mol==None :
            print(smiles)
        if self.stereo : isomers = list(EnumerateStereoisomers(mol))
        else : isomers = [mol]
        best_score = 10000
        best_smiles = ""
        for mol in isomers :
            s = Chem.MolToSmiles(mol, isomericSmiles=True)
            s = s+'Q'
            x = torch.tensor([self.c_to_i[i] for i in list(s)]).unsqueeze(0).to(device)
            output = self.model(x)
            p_char = nn.LogSoftmax(dim = -1)(output) * -1   #-log(probability)
            p_char = p_char.data.cpu().numpy()
            x = x.data.cpu().numpy()
            result = 0
            for i in range(output.size(1)-1):
                result += p_char[0, i, x[0, i]]
            if self.reduction == 'mean' :
                result /= output.size(1)-1
            if result < best_score :
                best_score = result
                best_smiles = s
        return best_score

    def normalize(self, score) :
        nondrug_score = self.nondrug_score  #It will be normalized to 0.0
        drug_score = self.drug_score        #It will be normalized to 1.0
        return (nondrug_score-score)/(nondrug_score - drug_score)

