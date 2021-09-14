import time
import numpy as np
import torch
import random
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
from models import RNNLM
import utils as UTILS
from utils.train_utils import train_without_validation_test
from utils.hydra_runner import hydra_runner

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

INPUT_SIZE = UTILS.DATA_UTILS.N_CHAR

@hydra_runner(config_path='config', config_name='rnn')
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
    model = RNNLM(model_params)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    #============ Load Train Data ===========#
    whole_data = UTILS.load_rnn_data(data_config)
    
    #============= Train ========#
    logging.info("============== Train Start ==============\n")
    train_without_validation_test(model, whole_data, train_params, data_config, save_file, device)
    logging.info(f"save file  : {save_file}\n")


if __name__ == '__main__' :
    main()
