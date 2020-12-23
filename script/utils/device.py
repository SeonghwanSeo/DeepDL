#The function for our work environment

import subprocess
import os
import torch

def set_cuda_visible_device(ngpus):
    """
    check empty node
    """
    empty = []
    for i in range(4):
        command = ['nvidia-smi','-i',str(i)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = str(p.communicate()[0])
        count = result.count('No running')
        if count>0:    
            empty.append(i)
    
    if len(empty)<ngpus:
        print (f'ERROR: avaliable gpus({len(empty)} gpus) are less than required({ngpus} gpus)')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device
