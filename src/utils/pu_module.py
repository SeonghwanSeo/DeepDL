import torch
import torch.nn as nn
import numpy as np
import logging

from .train_utils import train_with_5cv, train_with_validation_test

class PU_Fusilier_Module(nn.Module) :
    def __init__(self, model, whole_data, train_params, pu_params, save_file, device) :
        super(PU_Fusilier_Module, self).__init__()
        self.model = model
        self.th, self.max_iter = pu_params['threshold'], pu_params['max_iterations']
        self.save_file = save_file
        self.device = device
        self.train_params = train_params
        
        data, answer = whole_data
        pos_idx = np.where(answer == 1)[0]
        neg_idx = np.where(answer == 0)[0]
        self.pos_data = data[pos_idx]
        self.neg_data = data[neg_idx]
        self.real_neg_idx = np.arange(len(self.neg_data))
        self.unlabeled_idx = np.array([], dtype='i')
        self.len_pos = len(self.pos_data)

    def pu_train(self) :
        data = np.append(self.pos_data, self.neg_data[self.real_neg_idx])
        answer = np.array([1 for _ in range(self.len_pos)] + [0 for _ in range(len(self.real_neg_idx))])
        if self.train_params['val_mode'] == '5cv' :
            train_with_5cv(self.model, (data, answer), self.train_params, self.save_file, self.device)
        else :
            train_with_validation_test(self.model, (data, answer), self.train_params, self.save_file, self.device)

    def pu_check(self) :
        batch_size, num_workers = self.train_params['batch_size'], self.train_params['num_workers']
        real_neg_data = self.neg_data[self.real_neg_idx]
        answer = np.zeros(len(real_neg_data))
        _, val_dl = self.model.construct_dataloader(None, (real_neg_data, answer), batch_size, num_workers, None)
        y_pred = self.model.test_samples(val_dl)
        return np.where(y_pred<self.th)[0], np.where(y_pred>=self.th)[0]

    def run(self) :
        for _ in range(self.max_iter) :
            self.pu_train()
            real, unlabeled = self.pu_check()
            self.unlabeled_idx = np.append(self.unlabeled_idx, self.real_neg_idx[unlabeled])
            self.real_neg_idx = self.real_neg_idx[real]
            logging.info(f'Realiable Negative: {len(self.real_neg_idx)}\tUnlabeled: {len(self.unlabeled_idx)}\n')
            if len(unlabeled) == 0 :
                break
        return self.neg_data[self.real_neg_idx]

class PU_Liu_Module(nn.Module) :
    def __init__(self, model, whole_data, train_params, pu_params, save_file, device) :
        super(PU_Liu_Module, self).__init__()
        self.model = model
        self.th, self.max_iter = pu_params['threshold'], pu_params['max_iterations']
        self.save_file = save_file
        self.device = device
        self.train_params = train_params
        
        data, answer = whole_data
        pos_idx = np.where(answer == 1)[0]
        neg_idx = np.where(answer == 0)[0]
        self.pos_data = data[pos_idx]
        self.neg_data = data[neg_idx]
        self.real_neg_idx = np.arange(len(self.neg_data))
        self.unlabeled_idx = np.arange(len(self.neg_data))
        self.len_pos = len(self.pos_data)

    def pu_train(self) :
        data = np.append(self.pos_data, self.neg_data[self.real_neg_idx])
        answer = np.array([1 for _ in range(self.len_pos)] + [0 for _ in range(len(self.real_neg_idx))])
        if self.train_params['val_mode'] == '5cv' :
            train_with_5cv(self.model, (data, answer), self.train_params, self.save_file, self.device)
        else :
            train_with_validation_test(self.model, (data, answer), self.train_params, self.save_file, self.device)

    def pu_check(self) :
        batch_size, num_workers = self.train_params['batch_size'], self.train_params['num_workers']
        unlabeled_data = self.neg_data[self.unlabeled_idx]
        answer = np.zeros(len(unlabeled_data))
        _, val_dl = self.model.construct_dataloader(None, (unlabeled_data, answer), batch_size, num_workers, None)
        y_pred = self.model.test_samples(val_dl)
        return np.where(y_pred<self.th)[0], np.where(y_pred>=self.th)[0]

    def run(self) :
        self.pu_train()
        real, unlabeled = self.pu_check()
        self.real_neg_idx = self.unlabeled_idx[real]
        self.unlabeled_idx = self.unlabeled_idx[unlabeled]
        logging.info(f'Realiable Negative: {len(self.real_neg_idx)}\tUnlabeled: {len(self.unlabeled_idx)}\n')
        if len(real) == 0 :
            return self.neg_data[self.real_neg_idx]
        for _ in range(self.max_iter) :
            self.pu_train()
            real, unlabeled = self.pu_check()
            self.real_neg_idx = np.append(self.real_neg_idx, self.unlabeled_idx[real])
            self.unlabeled_idx = self.unlabeled_idx[unlabeled]
            logging.info(f'Realiable Negative: {len(self.real_neg_idx)}\tUnlabeled: {len(self.unlabeled_idx)}\n')
            if len(real) == 0 :
                break
        return self.neg_data[self.real_neg_idx]
