import numpy as np
import torch
from rdkit import Chem
import torch.nn as nn
import math
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import random

class logger() :
    def __init__(self, log_file) :
        self.log_file = log_file
        self.log=[]
        try :
            with open(self.log_file, 'w') as w :
                w.write('logging...')
            print("==========================================")
            print(f"start logging\n")
        except :
            print("ERR : Unvalid argument 'save_dir'")
            exit(1)

    def __call__(self, *log, ERR = False, put_log = True) :
        log_message = ' '.join([str(l) for l in log])
        print(log_message)
        if put_log :
            self.log.append(log_message)
        if ERR :
            self.save(True)
            exit(1)
            

    def save(self, ERR = False) :
        with open(self.log_file, 'w') as w :
            for log in self.log :
                w.write(log+'\n')
        if ERR :
            print("## There is ERROR MESSAGE ##")
            print(f"log finish")
            print(f"log file : {self.log_file}")

def splitting_data(train_data, validation_mode) :
    if validation_mode == 'val' :
        random.shuffle(train_data)
        split_idx = len(train_data)//5
        valid_data = train_data[:split_idx]
        train_data = train_data[split_idx:]
        return train_data, valid_data 
    elif validation_mode == '5cv' : 
        len_data = len(train_data)
        random.shuffle(train_data)
        train_data_list = [train_data[((len_data*i)//5):((len_data*(i+1))//5)] for i in range(5)]
        return train_data_list

def len_mask(l, max_length) :
    #l : tensor for length
    device = l.device
    mask = torch.arange(0, max_length).repeat(l.size(0)).to(device)
    l = l.unsqueeze(1).repeat(1, max_length).reshape(-1)
    mask = mask-l
    mask[mask>=0] = 0
    mask[mask<0] = 1 
    return mask

def convert_to_onehot(matrix, min_d, max_d, scaling):
    matrix[matrix>max_d] = max_d
    matrix[matrix<min_d] = min_d
    matrix -= min_d
    matrix /= scaling
    n_classes = int((max_d-min_d)/scaling+1)
    matrix = np.around(matrix).astype(int)
    retval = np.eye(n_classes)[matrix]    
    return retval

def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(4):
        command = ['nvidia-smi','-i',str(i)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = str(p.communicate()[0])
        count = result.count('No running')
        if count>0:    
            empty.append(i)
    
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=None):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
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

