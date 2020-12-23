import torch
from rdkit import Chem
import random
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

def save_model(model, save_file) :
    """
    save_model()
    save_model() works the same as function torch.save().
    Due to torch.save() save the Paralleled model(nn.DataParallel) when we trained the model in multi-gpu environment, we define save_model().
    """

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), save_file)
    else:
       torch.save(model.state_dict(), save_file)
    
def split_data(data, answer, validation_mode, shuffle) :
    """
    separate training part and validation part from train data
    In validation mode, we use 80% data for training and 20% data for validation test.    
    """
    if validation_mode == 'val': 
        if answer is None:
            train_data, valid_data = train_test_split(data, test_size=0.20, shuffle=shuffle, random_state=42)
        else :
            train_data, valid_data, _, _ = train_test_split(data, answer, test_size=0.20, stratify=answer, shuffle=shuffle, random_state=42)
        return train_data, valid_data

    elif validation_mode == '5cv':
        if answer is None:
            random_state = None
            if shuffle:
                random_state = 42
            kf = KFold(n_splits=5, shuffle=shuffle, random_state=random_state)
            return kf.split(data)
        else:
            random_state = None
            if shuffle:
                random_state = 42
            kf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=random_state)
            return kf.split(data, answer)

def len_mask(l, max_length) :
    """
    Mask the padding part of the result
    
    example data:
    c1ccccc1Q_________
    c1ccccc1Cc2ccccc2Q
    CNQ_______________
    ...

    We set the value of Padding part to 0
    """
    device = l.device
    mask = torch.arange(0, max_length).repeat(l.size(0)).to(device)
    l = l.unsqueeze(1).repeat(1, max_length).reshape(-1)
    mask = mask-l
    mask[mask>=0] = 0
    mask[mask<0] = 1 
    return mask

