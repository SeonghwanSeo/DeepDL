from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

random.seed(0)

class MolDataset(Dataset):
    def __init__(self, smiles, c_to_i, stereo=False):
        self.smiles = smiles
        self.c_to_i = c_to_i
        self.stereo = stereo

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        keydata=self.smiles[idx]
        if self.stereo :
            isomers=list(EnumerateStereoisomers(Chem.MolFromSmiles(keydata)))
            keydata = Chem.MolToSmiles(isomers[random.randint(0, len(isomers)-1)], isomericSmiles=True)
        else :
            keydata=Chem.MolToSmiles(Chem.MolFromSmiles(keydata))
        keydata += 'Q'
        sample = dict()
        sample['X'] = torch.from_numpy(np.array([self.c_to_i[c] for c in keydata]))
        sample['L'] = len(keydata)-1
        sample['n_char'] = len(self.c_to_i)
        sample['smiles'] = self.smiles[idx] 
        return sample


def my_collate(batch) :
    sample = dict()
    n_char = batch[0]['n_char']
    X = torch.nn.utils.rnn.pad_sequence([b['X'] for b in batch], batch_first=True, padding_value = n_char-1)
    L = torch.Tensor([b['L'] for b in batch])
    S = [b['smiles'] for b in batch]
    sample['X'] = X
    sample['L'] = L
    sample['S'] = S
    return sample
