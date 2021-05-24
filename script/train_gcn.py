import time
import numpy as np
import torch
import random
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import yaml

from models import GCNModel
import dataset as DATASET
import utils as UTILS
import arguments as ARGUMENTS

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

def main(args) : 
    #============ Make directory for save file =============#
    save_dir = '/'.join(args.save_file.split('/')[:-1])
    try :
        if os.path.isdir(save_dir) == False:
            os.system('mkdir '+save_dir)
    except :
        print('INACCESSIBLE SAVE FILE PATH(save_file): ' + args.save_file)
    
    #============ Load config ===============#
    with open(args.config, 'r') as f :
        config = yaml.safe_load(f)
        data_config = config['data_config']
        model_params = config['model_parameters']
        train_params = config['train_parameters']
    
    #============ Setup ============#
    setup_hparams(args, data_config, model_params, train_params)

    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(args.ngpu)

    #============ Construct model ==============#
    model = GCNModel(model_params)
    print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()

    #============ Load Train Data ===========#
    whole_data = UTILS.load_gcn_data(data_config)
    
    #============= Train ========#
    print("============== Train Start ==============\n")
    if train_params['val_mode'] == '5cv' :
        train_model_5cv(model, whole_data, train_params, data_config, args, device)
    else :
        train_model(model, whole_data, train_params, data_config, args, device)
    print(f"save file  : {args.save_file}\n")

#============= Setup hyperparameter ==================#
def setup_hparams(args, data_config, model_params, train_params) :
    #============ Set input_size  ===========#
    model_params['input_size'] = UTILS.DATA_UTILS.N_ATOM_FEATURE

    #============ Print arguments =============#
    arg_keys = ['ngpu', 'num_workers', 'batch_size', 'epoch', 'config', 'save_file']

    print('Arguments:')
    arg_dict = args.__dict__
    UTILS.pretty_print(arg_dict, dic_keys=arg_keys, indent = '\t', name_area_length = 14)
    print()
    
    #============ Print parameters ==============#
    d_keys = ['positive_file', 'negative_file', 'shuffle']
    m_keys = ['input_size', 'hidden_size1', 'hidden_size2', 'n_layers', 'n_heads', 'dropout']
    t_keys = ['init_model', 'val_mode', 'lr', 'lr_decay']

    print('Config')
    print("data_config:")
    UTILS.pretty_print(data_config, dic_keys=d_keys, indent = '\t', name_area_length=14)
    print()
    
    print("model_parameters:")
    UTILS.pretty_print(model_params, dic_keys=m_keys, indent='\t', name_area_length=14)
    print()
    
    print("train_parameters:")
    UTILS.pretty_print(train_params, dic_keys=t_keys, indent='\t', name_area_length=14)
    print()

#============= Construct dataloader =============#
def construct_dataloader(train_data, val_data, data_config, batch_size, num_workers) :
    train_dataset = DATASET.GraphDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, \
                                  num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    if val_data is not None :
        val_dataset = DATASET.GraphDataset(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, \
                                    num_workers=num_workers, collate_fn=val_dataset.collate_fn)
    else :
        val_dataloader = None

    return train_dataloader, val_dataloader

#============= Train model with a train/val data pair ========================#
def train_model(model, whole_data, train_params, data_config, args, device):
    answer = [int(label) for _, label in whole_data]
    shuffle = data_config['shuffle']
    train_data, val_data, _, _ = train_test_split(whole_data, answer, test_size = 0.20, shuffle=shuffle, \
                                                  random_state=42, stratify = answer)

    print(f"number of train_set : {len(train_data)}")
    print(f"number of val_set   : {len(val_data)}")
    print()
    
    #============ Construct Dataloader ===========#
    train_dataloader, val_dataloader = construct_dataloader(train_data, val_data, data_config, args.batch_size, args.num_workers)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params['init_model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    print("epoch | tloss  | vloss  | time\n")
    
    min_val_loss = 10000
    for epoch in range(args.epoch) :
        st = time.time()

        train_loss, val_loss = _train_epoch(model, optimizer, train_dataloader, val_dataloader)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)

        if val_loss < min_val_loss :
            min_val_loss = val_loss
            model.save_model(args.save_file)
        end = time.time()
        if False or epoch % 10 == 0 or epoch == args.epoch - 1 :
            print(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}\n")

#============= Train model with a 5cv mode ========================#
def train_model_5cv(model, whole_data, train_params, data_config, args, device):
    answer = [int(label) for _, label in whole_data]
    shuffle = data_config['shuffle']
    kf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42)

    print('5-fold cross validation test') 
    print(f"number of train_set : {len(whole_data)}")
    
    fold = 0
    max_epoch = args.epoch
    total_val_loss_list = [0 for _ in range(max_epoch)]
    overfit_cnt = 0

    for train_idx, val_idx in kf.split(whole_data, answer):
        fold+=1
        train_data, val_data = whole_data[train_idx], whole_data[val_idx]
        print(f"\n============== Fold {fold}/5 Start ==============\n")
        train_dataloader, val_dataloader = construct_dataloader(train_data, val_data, data_config, args.batch_size, args.num_workers)
        model = model.initialize_model(device, train_params['init_model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
        print("epoch | tloss  | vloss  | time\n")
        min_val_loss = 10000
        for epoch in range(max_epoch) :
            st = time.time()
            train_loss, val_loss = _train_epoch(model, optimizer, train_dataloader, val_dataloader)
            total_val_loss_list[epoch] += val_loss

            for param_group in optimizer.param_groups :
                param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)

            if val_loss < min_val_loss :
                overfit_cnt = 0
                min_val_loss = val_loss
            else :
                overfit_cnt += 1

            end = time.time()
            if False or epoch % 10 == 0 or overfit_cnt == 30 or epoch == max_epoch - 1 :
                print(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}\n")

            if overfit_cnt == 30 :
                break

        max_epoch = epoch + 1
        total_val_loss_list = total_val_loss_list[:max_epoch]

    #============ Calculate best epoch =============#
    print(f"\n============== All Fold End ==============\n")
    best_epoch = np.argmin(np.array(total_val_loss_list))
    print(f"best epoch: {best_epoch}")

    #============ Construct Dataloader with whole data ===========#
    train_dataloader, _ = construct_dataloader(whole_data, None, data_config, args.batch_size, args.num_workers)    
    
    print("\n============== Train Start ===============\n")
    #============ Train model ===============#
    model = model.initialize_model(device, train_params['init_model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    print("epoch | tloss  | vloss  | time\n")
    
    for epoch in range(best_epoch) :
        st = time.time()
        train_loss, _ = _train_epoch(model, optimizer, train_dataloader, None)
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)
    
        end = time.time()
        if False or epoch % 10 == 0 or epoch == best_epoch - 1 :
            print(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}\n")
    model.save_model(args.save_file)

def _train_epoch(model, optimizer, train_dataloader, val_dataloader) :
    train_loss_list = []
    val_loss_list = []
    model.train()
    for sample in train_dataloader :
        train_loss = model.train_step(sample, optimizer, gradient_clip_val = 1.0) 
        train_loss_list.append(train_loss.data.cpu().numpy())
    
    if val_dataloader is not None :
        model.eval()
        for sample in val_dataloader :
            val_loss = model.val_step(sample) 
            val_loss_list.append(val_loss.data.cpu().numpy())
    else :
        val_loss_list.append(0)

    train_loss = np.mean(np.array(train_loss_list))
    val_loss = np.mean(np.array(val_loss_list))
   
    return train_loss, val_loss 

if __name__ == '__main__' :
    args = ARGUMENTS.parser('gcn')
    main(args)
