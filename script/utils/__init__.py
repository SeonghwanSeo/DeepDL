import torch
import torch.nn as nn
import numpy as np
import random
from .device import *
from . import data_utils as DATA_UTILS

def load_data(model, args, data_config):
    #model: both model class and model object
    modeltype = model.modeltype
    datatype = model.require_datatype
    if 'shuffle' not in data_config:
        data_config['shuffle'] = True
    
    if modeltype == 'OneClassModel':
        answer = None
        if 'positive_file' not in data_config.keys() :
            print("""ERR: NO TRAIN FILE at data_config\ndata_config:\n\tpositive_file: None""")
            exit(-1)
        if 'stereo' not in data_config.keys() :
            data_config['stereo'] = True

        with open(data_config['positive_file']) as f : 
            train_data = f.readlines()
            train_data = [s.strip().split('\t')[-1] for s in train_data]
        input_size = DATA_UTILS.N_CHAR
        d_keys = ['positive_file', 'negative_file', 'stereo', 'shuffle'] 

    if modeltype == 'TwoClassModel':
            
        if 'positive_file' not in data_config.keys() or 'negative_file' not in data_config.keys() :
            print("""ERR: NO TRAIN FILE at data_config.\nPlease Write with this format\ndata_config:\n\tpositive_file: (File Name)\n\tnegative_file: (File Name)\n\tshuffle: True / False (default: True)\n\tbalance: True / False (default: False) If balance==True, we use the same amount of positive/negative data""")
            exit(-1)

        if 'balance' not in data_config.keys():
            data_config['balance'] = False
        if 'seed' in data_config.keys():
            random.seed(data_config['seed'])
        with open(data_config['positive_file']) as f :
            pos_file = f.readlines()
        with open(data_config['negative_file']) as f :
            neg_file = f.readlines()
        
        if data_config['balance']:
            if data_config['shuffle'] :
                random.shuffle(pos_file) 
                random.shuffle(neg_file) 
            pos_file = pos_file[:min(len(pos_file), len(neg_file))]
            neg_file = neg_file[:min(len(pos_file), len(neg_file))]
        
        train_data = []
        for smiles in pos_file:
            train_data.append((smiles.strip().split('\t')[-1], 1))
        for smiles in neg_file:
            train_data.append((smiles.strip().split('\t')[-1], 0))
        
        answer = []
        for _, label in train_data:
            answer.append(label)

        input_size = DATA_UTILS.N_ATOM_FEATURE
        d_keys = ['positive_file', 'negative_file', 'shuffle', 'balance'] 

    if answer :
        answer = np.array(answer)

    return np.array(train_data), answer, d_keys, input_size 
        
        
def initialize_model(model, device, load_save_file=None):
    """
    To set device of model
    If there is save_file for model, load it
    """
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file), strict=False) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model

def pretty_print(dic, dic_keys = None, org_keys=None, indent = '\t', name_area_length = 12) :
    """
    print dictionary.
    if dic_keys!=None, print only the components in dic_keys
    if org_dic!=None, label 'default' to each component when it is not in org_dic.
    """
    default_length = 12
    name_area_length = max(name_area_length, default_length)
    if dic_keys == None :
        dic_keys = list(dic.keys())
        dic_keys.sort()
    if org_keys == None :
        org_keys = dic_keys
    for k in dic_keys :
        if k in org_keys :
            print(f"{indent}{k:<{name_area_length}}: {dic[k]}")
        elif k in dic.keys() :
            print(f"{indent}{k:<{name_area_length}}: {dic[k]} (default)") 
