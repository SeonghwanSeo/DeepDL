import torch
import torch.nn as nn
import numpy as np
import random
from .device import *
from . import data_utils as DATA_UTILS

def load_rnn_data(data_config):
    #model: both model class and model object
    with open(data_config['positive_file']) as f : 
        train_data = f.readlines()
        train_data = [s.strip().split('\t')[-1] for s in train_data]
    return np.array(train_data)

def load_gcn_data(data_config):
    with open(data_config['positive_file']) as f :
        pos_file = f.readlines()
    with open(data_config['negative_file']) as f :
        neg_file = f.readlines()
    
    if data_config['shuffle'] :
        random.shuffle(pos_file) 
        random.shuffle(neg_file) 
    
    train_data = []
    for smiles in pos_file:
        train_data.append((smiles.strip().split('\t')[-1], 1))
    for smiles in neg_file:
        train_data.append((smiles.strip().split('\t')[-1], 0))
        
    return np.array(train_data)

def pretty_print(dic, dic_keys = None, indent = '\t', name_area_length = 12) :
    default_length = 12
    name_area_length = max(name_area_length, default_length)
    if dic_keys == None :
        dic_keys = list(dic.keys())
        dic_keys.sort()
    for k in dic_keys :
        print(f"{indent}{k:<{name_area_length}}: {dic[k]}")
