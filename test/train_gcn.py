import time
import numpy as np
import torch
import random
import sys
import os
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
from models import GCNModel
import utils as UTILS
from utils.train_utils import train_with_validation_test, train_with_5cv
from utils.hydra_runner import hydra_runner

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

INPUT_SIZE = UTILS.DATA_UTILS.N_ATOM_FEATURE

@hydra_runner(config_path='config', config_name='gcn')
def main(cfg) : 
    #============ Load config ===============#
    data_config = cfg.data
    model_params = cfg.model
    train_params = cfg.train
    
    #============ Setup ============#
    model_params.input_size = INPUT_SIZE
    save_dir, save_file = UTILS.exp_manager(cfg)

    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(train_params.gpus)

    #============ Construct model ==============#
    model = GCNModel(model_params)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    #============ Load Train Data ===========#
    whole_data = UTILS.load_gcn_data(data_config)
    
    #============= Train ========#
    if train_params['val_mode'] == '5cv' :
        train_with_5cv(model, whole_data, train_params, save_file, device)
    else :
        train_with_validation_test(model, whole_data, train_params, save_file, device)
    logging.info(f"save file  : {save_file}\n")

if __name__ == '__main__' :
    main()
