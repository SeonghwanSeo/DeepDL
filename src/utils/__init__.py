import torch
import torch.nn as nn
import numpy as np
import subprocess
import os
import random
import logging
from pathlib import Path
from omegaconf import OmegaConf

from . import data_utils as DATA_UTILS
from .loader import *

def exp_manager(cfg, exp_dir='result') :
    save_dir = os.path.join(exp_dir, cfg.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = os.path.join(save_dir, 'save.pt')
    log_file = os.path.join(save_dir, 'output.log')
    conf_file = os.path.join(save_dir, 'config.yaml')
    
    filehandler = logging.FileHandler(log_file, 'w')
    logger = logging.getLogger()
    logger.addHandler(filehandler)

    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    with open(conf_file, 'w') as w :
        OmegaConf.save(config=cfg, f=w, resolve=True)

    return save_dir, save_file

def set_cuda_visible_device(ngpus):
    """
    The function for our work environment
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
