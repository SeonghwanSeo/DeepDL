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
from utils.pu_module import PU_Fusilier_Module, PU_Liu_Module
from utils.hydra_runner import hydra_runner

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

INPUT_SIZE = UTILS.DATA_UTILS.N_ATOM_FEATURE

@hydra_runner(config_path='config', config_name='gcn_pu')
def main(cfg) : 
    #============ Load config ===============#
    data_config = cfg.data
    model_params = cfg.model
    train_params = cfg.train
    pu_params = cfg.pu_learning
    
    #============ Setup ============#
    model_params.input_size = INPUT_SIZE
    save_dir, _ = UTILS.exp_manager(cfg)

    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(train_params.gpus)

    #============ Construct model ==============#
    model = GCNModel(model_params)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    #============ Load Train Data ===========#
    whole_data = UTILS.load_gcn_data(data_config)
    
    #============= PU_Learning ========#
    assert pu_params.architecture in ['Fusilier', 'Liu']
    if pu_params.architecture == 'Fusilier' :
        pu_module = PU_Fusilier_Module(model, whole_data, train_params, pu_params, save_dir, device)
    elif pu_params.architecture == 'Liu' :
        pu_module = PU_Liu_Module(model, whole_data, train_params, pu_params, save_dir, device)
    pu_module.run()

if __name__ == '__main__' :
    main()
