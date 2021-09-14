import torch
import numpy as np
import time
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold

#============= Train model with a train/val data pair ========================#
def train_with_validation_test(model, whole_data, train_params, save_file, device):
    data, answer = whole_data
    X_train, X_val, Y_train, Y_val = train_test_split(data, answer, test_size = 0.20, random_state=42, stratify = answer)

    logging.info(f"number of train_set : {len(X_train)}")
    logging.info(f"number of val_set   : {len(X_val)}\n")
    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    
    batch_size = train_params.batch_size
    num_workers = train_params.num_workers
    max_epoch = train_params.epoch
    sampler = train_params.sampler
    #============ Construct Dataloader ===========#
    train_dataloader, val_dataloader = model.construct_dataloader(train_data, val_data, batch_size, num_workers, sampler)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params.init_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    logging.info("============== Train Start ==============\n")
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
    logging.info("\n============== Train End ==============\n")

#============= Train model with a 5cv mode ========================#
def train_with_5cv(model, whole_data, train_params, save_file, device):
    data, answer = whole_data[0].copy(), whole_data[1].copy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logging.info('5-fold cross validation test') 
    logging.info(f"number of train_set : {len(data)}\n")
    
    batch_size = train_params.batch_size
    num_workers = train_params.num_workers
    max_epoch = train_params.epoch
    sampler = train_params.sampler
    
    fold = 1
    max_epoch = train_params.epoch
    total_val_loss_list = [0 for _ in range(max_epoch)]
    overfit_cnt = 0

    logging.info("============== Train Start ==============")
    for train_idx, val_idx in kf.split(data, answer):
        logging.info(f"\n============== Fold {fold}/5 Start ==============\n")
        #============ Construct Dataloader ===========#
        train_data = data[train_idx], answer[train_idx]
        val_data = data[val_idx], answer[val_idx]
        train_dataloader, val_dataloader = model.construct_dataloader(train_data, val_data, batch_size, num_workers, sampler)

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
    train_dataloader, _ = model.construct_dataloader(whole_data, None, batch_size, num_workers, sampler)
    
    logging.info("============== Train Start ===============\n")
    #============ Train model ===============#
    model = model.initialize_model(device, train_params.init_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    logging.info("epoch | tloss  | time")
    
    for epoch in range(best_epoch) :
        st = time.time()
        train_loss, _ = model.train_epoch(train_dataloader, None, optimizer, gradient_clip_val=1.0)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)
    
        end = time.time()
        logging.info(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}")
    logging.info("\n============== Train End ==============\n")
    model.save_model(save_file)

#============= Train model with a train/val data pair ========================#
def train_without_validation_test(model, whole_data, train_params, data_config, save_file, device):
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
    sampler = train_params.sampler
    train_dataloader, _ = model.construct_dataloader(train_data, None, batch_size, num_workers, sampler)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params.init_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    logging.info("============== Train Start ==============\n")
    logging.info("epoch | tloss  | time")
    
    for epoch in range(max_epoch) :
        st = time.time()
        train_loss, _ = model.train_epoch(train_dataloader, None, optimizer, gradient_clip_val=1.0)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params.lr * (train_params.lr_decay ** epoch)
        
        end = time.time()
        logging.info(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}")
    
    logging.info("\n============== Train End ==============\n")
    model.save_model(save_file)
