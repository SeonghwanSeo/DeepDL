import numpy as np
import torch
from rdkit import Chem
import torch.nn as nn
import math
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

class logger() :
    def __init__(self, save_file) :
        self.log_file = save_file+'_log'
        self.log=[]
        try :
            with open(self.log_file, 'w') as w :
                w.write('')
            print("log -", self.log_file)
        except :
            print("ERR : Unvalid argument 'save_file'")
            exit(1)

    def __call__(self, log, put_log = True) :
        log = str(log)
        print(log)
        if put_log :
            self.log.append(log)

    def log(self) :
        with open(self.log_file, 'w') as w :
            for log in self.log :
                w.write(log+'\n')
        print("log finish -", self.log_file)

def distmaker(y,n,bend,send,out=False,norm=True):
    width=(bend-send)/n
    x=[send]+[send+width*(0.5+i) for i in range(n)]+[bend]
    count=[[] for i in range(n)]
    for i in y:
        if send<i<bend: count[int((i-send)/width)].append(y)
    yy=[0]+[len(i) for i in count]+[0]
    if norm:    return np.array(x),np.array(yy)/(width*len(y))
    return np.array(x),np.array(yy)

def mol_to_score(model, smiles, c_to_i, stereo=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = Chem.MolFromSmiles(smiles)
    if stereo : isomers = list(EnumerateStereoisomers(m))
    else : isomers = [m]
    best_score = 100
    for mol in isomers :
        s = Chem.MolToSmiles(mol, isomericSmiles=True)
        s = s+'Q'
        x = torch.tensor([c_to_i[i] for i in list(s)]).unsqueeze(0).to(device)
        output = model(x)
        p_char = nn.LogSoftmax(dim = -1)(output)
        p_char = p_char.data.cpu().numpy()
        x = x.data.cpu().numpy()
        result = 0
        for i in range(output.size(1)-1):
            result += p_char[0, i, x[0, i]]
        result = math.log(-result)
        best_score=min(best_score, result)
    return best_score

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

