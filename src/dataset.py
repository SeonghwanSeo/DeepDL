from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import utils.data_utils as DATA_UTILS

"""
We use various molecule representations, so we implement dataset class and collate function for each representation.
Each Dataset Class has static method collate_fn() for DataLoader.

Language model, RNN & Transformer
    SmilesDataset(): Class, Dataset class for SMILES

GAT model.
    GraphDataset(): Class, Dataset class for SMILES

Util functions or constant are implemented at 'utils/data_utils'
import Util as DATA_UTIL
"""
MAX_ATOM = 100
#=========== Dataset class for SMILES ============#
class SmilesDataset(Dataset):
    def __init__(self, smiles, stereo=True):
        self.smiles = smiles
        self.c_to_i = DATA_UTILS.C_TO_I
        self.stereo = stereo
        self.n_char = DATA_UTILS.N_CHAR

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
        sample['n_char'] = self.n_char
        sample['smiles'] = self.smiles[idx] 
        return sample

    @staticmethod 
    def collate_fn(batch) :
        sample = dict()
        n_char = batch[0]['n_char']
        X = torch.nn.utils.rnn.pad_sequence([b['X'] for b in batch], batch_first=True, padding_value = n_char-1)
        L = torch.Tensor([b['L'] for b in batch])
        S = [b['smiles'] for b in batch]
        sample['X'] = X
        sample['L'] = L
        sample['S'] = S
        return sample
#===============================================#

#============ Dataset class for Graph ==========#
class GraphDataset(Dataset):
    def __init__(self, data):
        self.feature_list = []
        self.num_atom_list = []
        self.adj_list = []
        smiles_list, label_list = data
        self.label_list = torch.from_numpy(label_list)
        self.__process(smiles_list)

    def __process(self, smiles_list):
        num_atom_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            n = mol.GetNumAtoms()
            adj = DATA_UTILS.get_adj(mol)
            feature = DATA_UTILS.get_atom_feature(mol)
            num_atom_list.append(n)
            self.adj_list.append(adj)
            self.feature_list.append(feature)
        self.num_atom_list = np.array(num_atom_list)

    def __getitem__(self,idx):
        sample = dict()
        sample['N'] = self.num_atom_list[idx]
        sample['A'] = self.adj_list[idx]
        sample['Y'] = self.label_list[idx]
        sample['F'] = self.feature_list[idx]
        return sample

    def __len__(self):
        return len(self.feature_list)

    @staticmethod 
    def collate_fn(batch):
        sample = dict()
        num_atom_list = []
        for b in batch:
            num_atom_list.append(b['N'])
        num_atom_list = np.array(num_atom_list)
        max_atom = MAX_ATOM
        adj_list = []
        af_list = []
        y_list = []
        for b in batch:
            k = b['N']
            padded_adj = np.zeros((max_atom,max_atom))
            padded_adj[:k,:k] = b['A']
            padded_feature = np.zeros((max_atom,31))
            padded_feature[:k,:31] = b['F']
            af_list.append(padded_feature)
            adj_list.append(padded_adj)
            y_list.append(b['Y'])
        sample['N'] = torch.Tensor(num_atom_list)
        sample['A'] = torch.Tensor(adj_list)
        sample['F'] = torch.Tensor(af_list)
        sample['Y'] = torch.Tensor(y_list)
        return sample

#===================================#
