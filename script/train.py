import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import os
import models
from rdkit import Chem
from torch.utils.data import DataLoader
import dataset as DATASET
import utils as UTILS
import utils.train_utils as TRAIN_UTILS
import yaml
import random
import copy
import sys
import arguments

def main(args) : 
    #============ Load Argument =============#
    ngpu = args.ngpu
    num_workers = args.num_workers
    batch_size = args.batch_size
    epoch = args.epoch
    save_file = args.save_file
    config = args.config

    save_dir = '/'.join(save_file.split('/')[:-1])
    try :
        if os.path.isdir(save_dir) == False:
            os.system('mkdir '+save_dir)
    except :
        print('INACCESSIBLE SAVE FILE PATH(save_file): ' + save_file)
    
    #============ Load config ===============#
    with open(config, 'r') as f :
        config = yaml.safe_load(f)
        data_config = config['data_config']
        model_params = config['model_parameters']
        train_params = config['train_parameters']
    d_keys_org = list(data_config.keys())
    m_keys_org = list(model_params.keys())
    t_keys_org = list(train_params.keys())

    model_type = model_params['model']
    if model_type not in models.MODEL_LIST.keys():
        print(f"""ERR: NOT ALLOWED MODEL TYPE '{model_type}'\nALLOWED MODEL TYPE: {list(models.MODEL_LIST.keys())}\nIf you add new type model, please add in './models/__init__.py'""")
        exit(1)
    
    train_params['bayesian'] = ('Bayesian' in model_type)   #hidden params

    model_class = models.MODEL_LIST[model_type]    #type: <class 'type'>. e.g. models.RNN

    #=========== Load Data ==============#
    whole_data, target, d_keys, input_size = UTILS.load_data(model_class, args, data_config)
    model_params['input_size'] = input_size

    #============ Set default value to omitted parameters ===========#
    if train_params['bayesian']:
        wc = 1e-4
        dc = 2.

        num_train = len(whole_data)*0.8
        model_params['wd'] = wc**2/num_train
        model_params['dd'] = dc/num_train
    """
    set omitted hyperparameters with default value.
    data_config(d_keys) is already set at "Load Data" section.
    model_params['input_size'] is already set at "Load Data" section
    """
    m_keys = model_class.initialize_params(model_params)
    
    t_keys = model_class.initialize_train_params(train_params)
    valid_mode = train_params['valid_mode']
    init_model = train_params['init_model']

    #============ Print arguments =============#
    print('Arguments:')
    arg_keys = ['ngpu', 'num_workers', 'batch_size', 'epoch', 'config', 'save_file']
    arg_dict = {'ngpu':ngpu, 'num_workers':num_workers, 'batch_size':batch_size, 'epoch':epoch, 'config':args.config, 'save_file':save_file}
    max_name_length = max(len(name) for name in arg_keys + d_keys + m_keys + t_keys)
    UTILS.pretty_print(arg_dict, dic_keys=arg_keys, indent = '\t', name_area_length = max_name_length)
    print()
    
    #============ Print parameters ==============#
    print('Config')
    print("data_config:")
    UTILS.pretty_print(data_config, dic_keys=d_keys, org_keys=d_keys_org, indent = '\t', name_area_length=max_name_length)
    print()
    
    print("model_parameters:")
    UTILS.pretty_print(model_params, dic_keys=m_keys, org_keys=m_keys_org, indent='\t', name_area_length=max_name_length)
    print()
    
    print("train_parameters:")
    UTILS.pretty_print(train_params, dic_keys=t_keys, org_keys=t_keys_org, indent='\t', name_area_length=max_name_length)
    print()
    
    del(m_keys_org)
    del(t_keys_org)
    
    #============ Set Device ============#
    device = UTILS.set_cuda_visible_device(args.ngpu)
    
    #============ Construct model ==============#
    model = model_class(model_params)
    print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()
    #============ Separate valid set from train set =====#
    """
    valid_mode: 'val': return value 'tv_set' is (train_set, valid_set), train_set: valid_set = 8:2
    valid_mode: '5cv': return value 'tv_set' is [set1, set2, set3, set4, set5], each set has 20% of whole_data
    """
    tv_set = TRAIN_UTILS.split_data(whole_data, target, valid_mode, data_config['shuffle'])

    #============= Train with normal validation ========#
    if valid_mode == 'val' :
        train_data, valid_data = tv_set

        print("we use normal validation test")
        print(f"number of train_set : {len(train_data)}")
        print(f"number of valid_set : {len(valid_data)}")
        print()
        print("============== Train Start ==============\n")
        _, best_epoch = _train_model(model, train_data, valid_data, data_config, train_params, epoch, batch_size, num_workers, save_file, device)
        print(f"best epoch : {best_epoch}")
        print(f"save file  : {save_file}\n")

    #============== Train with 5-fold cross-validation =============#
    elif valid_mode == '5cv' :
        fold=0
        fold_loss = []
        train_epoch = epoch
        if True:
            print("we use 5-fold cross validation test")
            print(f"number of train_set : {len(whole_data)}")
            print()
            for train_idx, valid_idx in tv_set :
                fold+=1
                train_data, valid_data = whole_data[train_idx], whole_data[valid_idx]
                print(f"============== Fold {fold} Start ==============\n")
                epoch_loss, _ = _train_model(model, train_data, valid_data, data_config, train_params, epoch, batch_size, num_workers, None, device)
                fold_loss.append(epoch_loss)
                epoch = min(epoch, len(epoch_loss))
        
            #============== calculate the train epoch ============= 
            print(f"============== All Fold End ==============\n")
            balanced_epoch = min([len(epoch_loss) for epoch_loss in fold_loss])      #to balance the size of each fold
            fold_loss = [epoch_loss[:balanced_epoch] for epoch_loss in fold_loss]
            fold_loss = np.array(fold_loss)
            fold_loss = np.mean(fold_loss, axis=0)
            train_epoch = np.argmin(fold_loss)+1
            print(f"best epoch : {train_epoch}\n")

        #============== train with calculated epoch ===========
        print("============== Train Start ===============\n")
        _train_model(model, whole_data, None, data_config, train_params, train_epoch, batch_size, num_workers, save_file, device)
        print("================  Finish  ================\n")
        print(f"best epoch : {train_epoch}")
        print(f"save file  : {save_file}\n")

#============= Train model with a train/valid data pair ========================"
def _train_model(model, train_data, valid_data, data_config, train_params, epoch, batch_size, num_workers, save_file, device):
    """
    There are two case for training
    i)  Training with valid check (valid_data != None)
    ii) Training w/o valid check (valid_data = None)   ==> When 5cv validation mode, we estimate the optimal epoch and train the model for that epoch.
    """
    #============ make dataset ==============# 
    model.init_cdropout()
    datatype = model.require_datatype
    train_dataset = make_dataset(train_data, datatype, data_config) 
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)
    if valid_data is not None:
        valid_dataset = make_dataset(valid_data, datatype, data_config) 
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=valid_dataset.collate_fn)
    
    #============ Train model ===============#
    model = UTILS.initialize_model(model, device, train_params['init_model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    min_loss = 1000000
    best_epoch = 0
    epoch_loss = []
    overfitting_cnt = 0         # to break the training after over-fitting. if the loss is bigger than min_loss for 50 times in a row, stop training.
    
    print("epoch | tloss  | vloss  | time\n")
    
    for epoch in range(1, epoch+1) :
        st = time.time()
        
        #train
        model.train()
        train_loss = run_model(model, train_dataloader, optimizer, device) 
        
        #validation
        if valid_data is not None:
            model.eval()
            with torch.no_grad():
                if train_params['bayesian'] :
                    valid_loss = np.mean(np.array([run_model(model,valid_dataloader, None, device) for _ in range(train_params.get('mc_sample', 20))]))
                else :
                    valid_loss = run_model(model, valid_dataloader, None, device)

            #find optimal model
            epoch_loss.append(valid_loss) 
            if valid_loss < min_loss :
                overfitting_cnt=0
                min_loss = valid_loss
                best_epoch = epoch
                if save_file :
                    TRAIN_UTILS.save_model(model, save_file)
            else :
                overfitting_cnt+=1
        end = time.time()
        if True or overfitting_cnt == 100 or epoch%10 == 0 :
            if valid_data is not None:
                print(f"{epoch:<5d} |{train_loss:7.4f} |{valid_loss:7.4f} | {end-st:.2f}\n")
            else :
                print(f"{epoch:<5d} |{train_loss:7.4f} |        | {end-st:.2f}\n")
 
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)

        if overfitting_cnt == 100 : break
    
    # final training of 5-fold cross-validation 
    if save_file and valid_data is None :  
        TRAIN_UTILS.save_model(model, save_file)
            
    return epoch_loss, best_epoch


#======== Run model for one dataloader =================#
"""
We use various datatype and various model type (one classification model or two classification model)
Since the model's learning method is determined by the relevant factors, it is implemented as an inner code according to each method.
If you want to add another type model, please add inner code.

run_model
    - _run_language_model()
    - _run_graph_model()
"""
def run_model(model, dataloader, optimizer, device) :
    """
    model: target model
    dataloader: dataloader (train_dataloader / valid_dataloader)
    optimizer: optimizer and the sign whether it is a training or testing section.
    device: CUDA or cpu.
    """
    if type(model).__name__ == 'DataParallel' :
        modeltype = model.module.modeltype
        datatype = model.module.require_datatype
    else : 
        modeltype = model.modeltype
        datatype = model.require_datatype
    if modeltype == 'OneClassModel' and datatype == 'SMILES' :
        return _run_language_model(model, dataloader, optimizer, device)
    elif modeltype == 'TwoClassModel' and datatype == 'Graph' :
        return _run_graph_model(model, dataloader, optimizer, device) 
    else :
        print("ERR: NOT ALLOWED MODELTYPE/DATATYPE. Check model.modeltype and model.require_datatype")

def _run_language_model(model, dataloader, optimizer, device) :
    loss_list = []
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    for i_batch, sample in enumerate(dataloader) :
        x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
        output = model(x)
        mask = TRAIN_UTILS.len_mask(l+1, output.size(1)-1)
        loss=torch.sum(loss_fn(output[:,:-1].reshape(-1, UTILS.DATA_UTILS.N_CHAR), x.reshape(-1))*mask)/mask.sum()
        loss_list.append(loss.data.cpu().numpy())
        if optimizer :
            loss.backward()    
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            model.zero_grad()
    return np.mean(np.array(loss_list)) 

def _run_graph_model(model, dataloader, optimizer, device) :
    total_loss = 0
    rr_list=[]
    loss_fn = nn.BCELoss(reduction='mean')
    training = (optimizer != None)
    for i_batch, sample in enumerate(dataloader) :
        x, adj, y = sample['F'].to(device).float(), sample['A'].to(device).float(), sample['Y'].to(device).float()
        y_pred = model(x, adj)   #x: atom_feature
        if training and getattr(model, 'regularisation', None) :
            loss = loss_fn(y_pred.squeeze(-1), y) + model.regularisation()
        else :
            loss = loss_fn(y_pred.squeeze(-1), y)
        if training :
            loss.backward()    
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            model.zero_grad()
        total_loss += float(loss.data.cpu().numpy()) * x.shape[0]
    total_loss /= len(dataloader.dataset)
    return total_loss

#============= Function for construct Dataset ============#
def make_dataset(data, datatype, data_config) :
    """
    combine Dataset and DataLoader
    extra_feature: For extra argument.
    """
    if datatype == 'SMILES' :
        stereo = data_config['stereo']
        dataset = DATASET.SmilesDataset(data, stereo)
    elif datatype == 'Graph' :
        dataset = DATASET.GraphDataset(data)
    else :
        print("ERR: NOT ALLOWED DATA TYPE")
        exit(-1)

    return dataset

if __name__ == '__main__' :
    args = arguments.parser()
    
    main(args)

