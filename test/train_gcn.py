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
from utils.hydra_runner import hydra_runner

seed = 5
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
    save_file = UTILS.exp_manager(cfg)

    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(train_params.gpus)

    #============ Construct model ==============#
    model = GCNModel(model_params)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    #============ Load Train Data ===========#
    whole_data = UTILS.load_gcn_data(data_config)
    
    #============= Train ========#
    logging.info("============== Train Start ==============\n")
    if train_params['val_mode'] == '5cv' :
        train_model_5cv(model, whole_data, train_params, save_file, device)
    else :
        train_model(model, whole_data, train_params, save_file, device)
    logging.info(f"save file  : {save_file}\n")

#============= Train model with a train/val data pair ========================#
def train_model(model, whole_data, train_params, save_file, device):
    answer = [int(label) for _, label in whole_data]
    train_data, val_data, _, _ = train_test_split(whole_data, answer, test_size = 0.20, stratify = answer)

    logging.info(f"number of train_set : {len(train_data)}")
    logging.info(f"number of val_set   : {len(val_data)}\n")
    
    batch_size = train_params.batch_size
    num_workers = train_params.num_workers
    max_epoch = train_params.epoch
    #============ Construct Dataloader ===========#
    train_dataloader, val_dataloader = model.construct_dataloader(train_data, val_data, batch_size, num_workers)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params.init_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    logging.info("epoch | tloss  | vloss  | time")
    min_val_loss = 10000
    for epoch in range(max_epoch) :
        st = time.time()
        train_loss, val_loss = model.train_epoch(train_dataloader, val_dataloader, optimizer, gradient_clip_val=1.0)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)

        if val_loss < min_val_loss :
            min_val_loss = val_loss
            model.save_model(save_file)

        end = time.time()
        logging.info(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}")

#============= Train model with a 5cv mode ========================#
def train_model_5cv(model, whole_data, train_params, save_file, device):
    answer = [int(label) for _, label in whole_data]
    kf = StratifiedKFold(n_splits=5)

    logging.info('5-fold cross validation test') 
    logging.info(f"number of train_set : {len(whole_data)}")
    
    batch_size = train_params.batch_size
    num_workers = train_params.num_workers
    max_epoch = train_params.epoch
    
    fold = 1
    max_epoch = train_params.epoch
    total_val_loss_list = [0 for _ in range(max_epoch)]
    overfit_cnt = 0

    for train_idx, val_idx in kf.split(whole_data, answer):
        logging.info(f"\n============== Fold {fold}/5 Start ==============\n")
        #============ Construct Dataloader ===========#
        train_data, val_data = whole_data[train_idx], whole_data[val_idx]
        train_dataloader, val_dataloader = model.construct_dataloader(train_data, val_data, batch_size, num_workers)

        #============ Train model ===============#
        model = model.initialize_model(device, train_params.init_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
        logging.info("epoch | tloss  | vloss  | time")
        min_val_loss = 10000
        for epoch in range(max_epoch) :
            st = time.time()
            train_loss, val_loss = model.train_epoch(train_dataloader, val_dataloader, optimizer, gradient_clip_val=1.0)
            total_val_loss_list[epoch] += val_loss

            for param_group in optimizer.param_groups :
                param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)

            if val_loss < min_val_loss :
                overfit_cnt = 0
                min_val_loss = val_loss
            else :
                overfit_cnt += 1

            end = time.time()
            logging.info(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}")

            if overfit_cnt == 100 :
                break

        max_epoch = epoch + 1
        fold += 1
        total_val_loss_list = total_val_loss_list[:max_epoch]

    #============ Calculate best epoch =============#
    logging.info(f"\n============== All Fold End ==============\n")
    best_epoch = np.argmin(np.array(total_val_loss_list))
    logging.info(f"best epoch: {best_epoch}\n")

    #============ Construct Dataloader with whole data ===========#
    train_dataloader, _ = model.construct_dataloader(whole_data, None, batch_size, num_workers)
    
    logging.info("============== Train Start ===============\n")
    #============ Train model ===============#
    model = model.initialize_model(device, train_params.init_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    logging.info("epoch | tloss  | time")
    
    for epoch in range(best_epoch) :
        st = time.time()
        for param_group in optimizer.param_groups :
            train_loss, _ = model.train_epoch(train_dataloader, None, optimizer, gradient_clip_val=1.0)
            param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)
    
        end = time.time()
        logging.info(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}")
    model.save_model(save_file)

if __name__ == '__main__' :
    main()
