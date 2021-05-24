import time
import numpy as np
import torch
import random
import os
from rdkit import Chem
from torch.utils.data import DataLoader
import yaml

from models import RNNLM
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
    model = RNNLM(model_params)
    print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()

    #============ Load Train Data ===========#
    whole_data = UTILS.load_rnn_data(data_config)
    
    #============= Train ========#
    print("============== Train Start ==============\n")
    train_model(model, whole_data, train_params, data_config, args, device)
    print(f"save file  : {args.save_file}\n")

#============= Setup hyperparameter ==================#
def setup_hparams(args, data_config, model_params, train_params) :
    #============ Set input_size  ===========#
    model_params['input_size'] = UTILS.DATA_UTILS.N_CHAR

    #============ Print arguments =============#
    arg_keys = ['ngpu', 'num_workers', 'batch_size', 'epoch', 'config', 'save_file']

    print('Arguments:')
    arg_dict = args.__dict__
    UTILS.pretty_print(arg_dict, dic_keys=arg_keys, indent = '\t', name_area_length = 14)
    print()
    
    #============ Print parameters ==============#
    d_keys = ['positive_file', 'stereo']
    m_keys = ['input_size', 'hidden_size', 'n_layers', 'dropout']
    t_keys = ['init_model', 'lr', 'lr_decay']
    
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
    train_dataset = DATASET.SmilesDataset(train_data, data_config['stereo'])
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, \
                                  num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    val_dataset = None
    val_dataloader = None

    return train_dataloader, val_dataloader

#============= Train model with a train/val data pair ========================#
def train_model(model, whole_data, train_params, data_config, args, device):
    """
    Unlike TCC, the definition of validation loss in distribution learning is ambiguous.
    Therefore, we do not perform validation test during training RNNLM.
    """
    train_data = whole_data
    print(f"number of train_set : {len(train_data)}")
    print()

    #============ Construct Dataloader ===========#
    train_dataloader, _ = construct_dataloader(train_data, None, data_config, args.batch_size, args.num_workers)
    
    #============ Train model ===============#
    model = model.initialize_model(device, train_params['init_model'])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    print("epoch |  loss  | time\n")
    
    for epoch in range(args.epoch) :
        st = time.time()
        train_loss_list = []
        for sample in train_dataloader :
            train_loss = model.train_step(sample, optimizer, gradient_clip_val = 1.0) 
            train_loss_list.append(train_loss.data.cpu().numpy())
        for param_group in optimizer.param_groups :
            param_group['lr'] = train_params['lr'] * (train_params['lr_decay'] ** epoch)
        
        train_loss = np.mean(np.array(train_loss_list))
        end = time.time()
        print(f"{epoch:<5d} |{train_loss:7.4f} | {end-st:.2f}\n")
    
    model.save_model(args.save_file)

if __name__ == '__main__' :
    args = ARGUMENTS.parser('rnn')
    main(args)
