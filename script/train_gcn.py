import time
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import yaml

from models import GCNModel
import dataset as DATASET
import utils as UTILS
import arguments as ARGUMENTS

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
    print(f"save file  : {save_file}\n")

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
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    if val_data is not None :
        val_dataset = DATASET.GraphDataset(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)

    return train_dataloader, val_dataloader

#============= Train model with a train/val data pair ========================#
def train_model(model, whole_data, train_params, data_config, args, device):
    answer = [int(label) for _, label in whole_data]
    train_data, val_data, _, _ = train_test_split(whole_data, answer, test_size = 0.20, shuffle=data_config['shuffle'], random_state=42, stratify = answer)

    print(f"number of train_set : {len(train_data)}")
    print(f"number of val_set   : {len(val_data)}")
    print()
    
    #============ Construct Dataloader ===========#
    train_dataloader, val_dataloader = construct_dataloader(train_data, val_data, data_config, args.batch_size, args.num_workers)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params['init_model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    print("epoch | tloss  | vloss  | time\n")
    
    for epoch in range(args.epoch) :
        st = time.time()
        train_loss_list = []
        val_loss_list = []
        min_val_loss = 10000
        model.train()
        for sample in train_dataloader :
            train_loss = model.train_step(sample, optimizer, gradient_clip_val = 1.0) 
            train_loss_list.append(train_loss.data.cpu().numpy())
        
        model.eval()
        for sample in val_dataloader :
            val_loss = model.val_step(sample) 
            val_loss_list.append(val_loss.data.cpu().numpy())

        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)
        
        train_loss = np.mean(np.array(train_loss_list))
        val_loss = np.mean(np.array(val_loss_list))

        if val_loss < min_val_loss :
            min_val_loss = val_loss
            model.save_model(args.save_file)
        end = time.time()
        print(f"{epoch:<5d} |{train_loss:7.4f} |{val_loss:7.4f} | {end-st:.2f}\n")

if __name__ == '__main__' :
    args = ARGUMENTS.parser('gcn')
    main(args)
