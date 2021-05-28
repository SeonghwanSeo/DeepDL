import time
import numpy as np
import torch
import random
import os
import logging

from models import RNNLM
import utils as UTILS
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
    save_file = UTILS.exp_manager(cfg)

    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(train_params.gpus)

    #============ Construct model ==============#
    model = RNNLM(model_params)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    #============ Load Train Data ===========#
    whole_data = UTILS.load_rnn_data(data_config)
    
    #============= Train ========#
    logging.info("============== Train Start ==============\n")
    train_model(model, whole_data, train_params, data_config, save_file, device)
    logging.info(f"save file  : {save_file}\n")

#============= Train model with a train/val data pair ========================#
def train_model(model, whole_data, train_params, data_config, save_file, device):
    """
    Unlike TCC, the definition of validation loss in distribution learning is ambiguous.
    Therefore, we do not perform validation test during training RNNLM.
    """
    train_data = whole_data
    logging.info(f"number of train_set : {len(train_data)}\n")

    #============ Construct Dataloader ===========#
    batch_size = train_params.batch_size
    num_workers = train_params.num_workers
    max_epoch = train_params.epoch
    train_dataloader, _ = model.construct_dataloader(train_data, None, batch_size, num_workers)
    
    #============ Train model ===============#
    init_model_file = os.path.join(train_params.init_model, 'save.pt')
    model = model.initialize_model(device, init_model_file)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    logging.info("epoch | tloss  | time")
    
    for epoch in range(max_epoch) :
        st = time.time()
        train_loss, _ = model.train_epoch(train_dataloader, None, optimizer, gradient_clip_val=1.0)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)
        
        end = time.time()
        logging.info(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}")
    
    model.save_model(save_file)

if __name__ == '__main__' :
    main()
