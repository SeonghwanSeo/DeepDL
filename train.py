import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from models.RNN import RNN
from models.Transformer import TransformerModel
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import MolDataset, my_collate
import utils
import yaml
from sklearn.model_selection import KFold
import random

parser = argparse.ArgumentParser()
parser.add_argument("--ngpu", help="number of gpu", type=int, default=4)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=400)
parser.add_argument("--epoch", help="epoch", type=int, default=100)
parser.add_argument("--save_file", help="save directory", type=str, default = './output')
parser.add_argument("--config", help="config file", type=str, default="config/pretrain.yaml")
args = parser.parse_args()
print(args)
print()

try :
    with open(args.save_file+'_save.pt', 'w') as w :
        w.write("training...")
except :
    print("ERR : Invalid argument 'save_file'")
    exit(1)

if args.ngpu>0:
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
else:
    os.environ['CUDA_VISIBLE_DEVICES']=''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#=========== loading data ===========
with open(args.config, 'r') as f :
    config = yaml.safe_load(f)
    data_config = config['data_config']
    model_params = config['model_params']

c_to_i = pickle.load(open(data_config['c_to_i'], 'rb'))
i_to_c = pickle.load(open(data_config['i_to_c'], 'rb'))
n_char = len(c_to_i)

#=========== model ===========
if model_params['kind']=='RNN' :
    model = RNN(model_params, n_char, i_to_c)
elif model_params['kind']=='Transformer':
    model = TransformerModel(model_params, n_char, i_to_c)
else :
    print("ERR : Not allowed model. Default model is RNN")
    exit(1)

print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print()
pretrain = data_config.get('pretrain', None)
if pretrain :
    print(f'pretrain : {pretrain}\n')

#============ def train function ==========
def model_train(train_data, valid_data, epoch, save_file = None, device = device, model = model, init_model = pretrain) : 
    train_dataset = MolDataset(train_data, dict(c_to_i), model_params['stereo'])
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
    if valid_data :
        valid_dataset = MolDataset(valid_data, dict(c_to_i), model_params['stereo'])
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=my_collate) 

    model = utils.initialize_model(model, device, init_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'])
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    min_loss = 10000
    min_epoch = 0
    epoch_loss = []
    overfitting_cnt = 0         # to break the training after over-fitting. if the loss is bigger than min_loss for 100 times in a row, stop training.
    
    if valid_data :
        print("epoch | tloss  | vloss  | time\n")
    else :
        print("epoch | tloss  | time\n")
        
    for epoch in range(1, epoch+1) :
        st = time.time()
        train_loss = []
        valid_loss = []
        
        #train
        model.train()
        for i_batch, sample in enumerate(train_dataloader) :
            x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
            output = model(x)
            mask = utils.len_mask(l+1, output.size(1)-1)
            loss=torch.sum(loss_fn(output[:,:-1].reshape(-1, n_char), x.reshape(-1))*mask)/mask.sum()
            loss.backward()    
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss.append(loss.data.cpu().numpy())
            model.zero_grad()
        
        #validation
        if valid_data :
            model.eval()
            with torch.no_grad():
                for i_batch, sample in enumerate(valid_dataloader) :
                    x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
                    output = model(x) 
                    mask = utils.len_mask(l+1, output.size(1)-1)
                    loss=torch.sum(loss_fn(output[:,:-1].reshape(-1, n_char), x.reshape(-1))*mask)/mask.sum()
                    valid_loss.append(loss.data.cpu().numpy())

            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            epoch_loss.append(valid_loss)
            
            if valid_loss < min_loss :
                overfitting_cnt=0
                min_loss = valid_loss
                min_epoch = epoch
                if save_file :
                    name = f'{save_file}_save.pt'
                    if args.ngpu>1:
                        torch.save(model.module.state_dict(), name)
                    else:
                        torch.save(model.state_dict(), name)
            else :
                overfitting_cnt+=1
                if overfitting_cnt == 100 : break

            for param_group in optimizer.param_groups :
                param_group['lr'] = model_params['lr'] * (model_params['lr_decay'] ** epoch)

            end = time.time()
            if epoch%10 == 0 : ##########################
                print(f"{epoch:<5d} |{train_loss:7.4f} |{valid_loss:7.4f} | {end-st:.2f}\n") 

        else :  # final training of 5-fold cross-validation 
            train_loss = np.mean(np.array(train_loss))
            end = time.time()
            if epoch%10 == 0 :
                print(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}\n") 

    if not valid_data and save_file :
        name = f'{save_file}_save.pt'
        if args.ngpu>1:
            torch.save(model.module.state_dict(), name)
        else:
            torch.save(model.state_dict(), name)
            
    return epoch_loss, min_epoch

#========== spliting data ==========
valid_mode = data_config['valid_mode']
 
if valid_mode=='val' :
    if data_config.get('valid_file', None) == None :
        print("ERR : There is no validation file(valid_file) at data config")
    with open(data_config['train_file']) as f:
        train_data = f.readlines()
        train_data = [s.strip().split('\t')[1] for s in train_data]
    with open(data_config['valid_file']) as f:
        valid_data = f.readlines()
        valid_data = [s.strip().split('\t')[1] for s in valid_data]
    print("we use validation test")
    print("number of train_set :", len(train_data))
    print("number of valid_set :", len(valid_data))

elif valid_mode=='5cv' :
    with open(data_config['train_file']) as f:
        train_data = f.readlines()
        train_data = [s.strip().split('\t')[1] for s in train_data]
    random.shuffle(train_data)
    train_data_list = [train_data[((len(train_data)*i)//5):((len(train_data)*(i+1))//5)] for i in range(5)]
    print("we use 5-fold cross-validation test")
    print("number of train_set :", len(train_data))
print()

#============== main code ========================
if valid_mode == 'val' :
    print("============== Train Start ==============\n")
    _, min_epoch = model_train(train_data, valid_data, epoch = args.epoch, save_file = args.save_file)
    print(f"best epoch : {min_epoch}\nsave file  : {args.save_file}_save.pt")
    print()

#============== 5-fold cross-validation =============
elif valid_mode == '5cv' :
    fold_loss = []
    min_epoch = args.epoch

    for fold in range(5) :
        train_data_ = []
        valid_data_ = []
        for i in range(5) :
            if i==fold :
                valid_data_+=train_data_list[i]
            else :
                train_data_+=train_data_list[i]
        print(f"============== Fold {fold} Start ==============\n")
        epoch_loss, _ = model_train(train_data_, valid_data_, epoch = args.epoch)
        fold_loss.append(epoch_loss)
        min_epoch = min(min_epoch, len(epoch_loss))
    
    #============== calculate the train epoch =============
     
    print(f"============== All Fold End ==============\n")
    fold_loss = [el[:min_epoch] for el in fold_loss]
    fold_loss = np.array(fold_loss)
    fold_loss = np.mean(fold_loss, axis=0)
    train_epoch = np.argmin(fold_loss)+1
    print(f"best epoch : {train_epoch}\n")

    #============== train with calculated epoch ===========
    print("============== Train Start ===============\n")
    model_train(train_data, None, epoch = train_epoch, save_file = args.save_file)
    print(f"best epoch : {train_epoch}\nsave file  : {args.save_file}_save.pt\n")


