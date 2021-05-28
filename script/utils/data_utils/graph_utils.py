import numpy as np 
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

#For Graph
symbol_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'X']
numH_list = [0, 1, 2, 3, 4, 5, 6]
degree_list = [0, 1, 2, 3, 4, 5, 6]
impval_list = [0, 1, 2, 3, 4, 5]

N_ATOM_FEATURE = len(symbol_list) + len(numH_list) + len(degree_list) + len(impval_list) + 1

def _one_hot_encoding(x, value_list):
    vector = [0] * len(value_list)
    if x in value_list:
        vector[value_list.index(x)] = 1
    else:
        vector[-1] = 1
    return vector

def _atom_feature(atom):
    symbol_vector = _one_hot_encoding(atom.GetSymbol(),symbol_list)
    numH_vector = _one_hot_encoding(atom.GetTotalNumHs(),numH_list)
    degree_vector = _one_hot_encoding(atom.GetDegree(),degree_list)
    impval_vector = _one_hot_encoding(atom.GetImplicitValence(), impval_list)
    aroma_vector = [atom.GetIsAromatic()]
    # 10 + 6 + 6+ 1
    tot_vector = symbol_vector + numH_vector + degree_vector + impval_vector + aroma_vector
    return np.array(tot_vector)

def get_atom_feature(mol):
    feature = []
    for atom in mol.GetAtoms():
        feature.append(_atom_feature(atom))
    return np.array(feature)
    
def get_adj(mol) :
    return GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
