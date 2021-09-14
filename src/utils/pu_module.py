import os
import torch
import torch.nn as nn
import numpy as np
import logging

from .train_utils import train_with_5cv, train_with_validation_test, train_without_validation_test

SAVE_SIGN = 0b01
BREAK_SIGN = 0b10
RUN = 0b01
FINISH = 0b11
FAIL = 0b10

class PU_Module(nn.Module) :
    def __init__(self) :
        super(PU_Module, self).__init__()
        self.model = None
        self.th = None
        self.max_iteration = None
        self.save_dir = None
        self.train_params = None
        self.device = None
        self.pos_data = None
        self.neg_data = None
        self.real_neg_idx = None
        self.unlabeled_idx = None

    def run(self) :
        for iteration in range(self.max_iteration) :
            sign = self.run_iteration(iteration)
            if sign ^ FAIL :
                logging.info(f'Realiable Negative: {len(self.real_neg_idx)}\tUnlabeled: {len(self.unlabeled_idx)}\n')
            if sign & SAVE_SIGN :
                self.save_iteration(iteration)
            if sign & BREAK_SIGN :
                break

    def run_iteration(self, iteration) -> bool :
        return False

    def train(self, pos_data, neg_data) :
        self.model.train()
        data = np.append(pos_data, neg_data)
        answer = np.array([1 for _ in range(len(pos_data))] + [0 for _ in range(len(neg_data))])
        if self.train_params['val_mode'] == '5cv' :
            train_with_5cv(self.model, (data, answer), self.train_params, None, self.device)
        else :
            train_with_validation_test(self.model, (data, answer), self.train_params, None, self.device)

    @torch.no_grad()
    def check(self, test_data) :
        self.model.eval()
        answer = np.zeros(len(test_data))
        batch_size, num_workers = self.train_params['batch_size'], self.train_params['num_workers']
        _, val_dl = self.model.construct_dataloader(None, (test_data, answer), batch_size, num_workers, None)
        y_pred = self.model.test_samples(val_dl)
        return np.where(y_pred<self.th)[0], np.where(y_pred>=self.th)[0] 

    def save_iteration(self, iteration) :
        iter_dir = os.path.join(self.save_dir, f'iteration_{iteration}')
        model_file = os.path.join(iter_dir, 'save.pt')
        smiles_file = os.path.join(iter_dir, 'negative.smi')
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        self.model.save_model(model_file)
        smiles_list = self.neg_data[self.real_neg_idx].tolist()
        with open(smiles_file, 'w') as w :
            w.write('\n'.join(smiles_list))

class PU_Fusilier_Module(PU_Module) :
    def __init__(self, model, whole_data, train_params, pu_params, save_dir, device) :
        super(PU_Fusilier_Module, self).__init__()
        self.model = model
        self.th, self.max_iteration = pu_params['threshold'], pu_params['max_iterations']
        self.save_dir = save_dir
        self.device = device
        self.train_params = train_params
        
        data, answer = whole_data
        pos_idx = np.where(answer == 1)[0]
        neg_idx = np.where(answer == 0)[0]
        self.pos_data = data[pos_idx]
        self.neg_data = data[neg_idx]
        self.n_pos = len(self.pos_data)
        self.n_neg = len(self.neg_data)
        self.real_neg_idx = np.arange(len(self.neg_data))
        self.unlabeled_idx = np.array([], dtype='i')
    
    def run_iteration(self, iteration) :
        real_data = self.neg_data[self.real_neg_idx]
        self.train(self.pos_data, real_data)
        real, unlabeled = self.check(real_data)
        n_real = len(real)
        if len(real) < self.n_pos :
            logging.info(f'The Number of Realiable Negative Set is lower than those of Positive Set. Use Right Before Iteration\n')
            return FAIL
        self.unlabeled_idx = np.append(self.unlabeled_idx, self.real_neg_idx[unlabeled])
        self.real_neg_idx = self.real_neg_idx[real]
        
        if len(unlabeled) == 0 :
            return FINISH
        else :
            return RUN

class PU_Liu_Module(PU_Module) :
    def __init__(self, model, whole_data, train_params, pu_params, save_dir, device) :
        super(PU_Liu_Module, self).__init__()
        self.model = model
        self.th, self.max_iteration = pu_params['threshold'], pu_params['max_iterations']
        self.save_dir = save_dir
        self.device = device
        self.train_params = train_params
        
        data, answer = whole_data
        pos_idx = np.where(answer == 1)[0]
        neg_idx = np.where(answer == 0)[0]
        self.pos_data = data[pos_idx]
        self.neg_data = data[neg_idx]
        self.real_neg_idx = None
        self.unlabeled_idx = np.arange(len(self.neg_data))
        self.len_pos = len(self.pos_data)

    def run_iteration(self, iteration) :
        if iteration == 0 :
            self.train(self.pos_data, self.neg_data)
            real, unlabeled = self.check(self.neg_data)
            self.real_neg_idx = self.unlabeled_idx[real]
            self.unlabeled_idx = self.unlabeled_idx[unlabeled]
            if len(real) == 0 :
                return FAIL
        else :
            real_neg_data = self.neg_data[self.real_neg_idx]
            unlabeled_data = self.neg_data[self.unlabeled_idx]
            self.train(self.pos_data, real_neg_data)
            real, unlabeled = self.pu_check(unlabeled_data)
            self.real_neg_idx = np.append(self.real_neg_idx, self.unlabeled_idx[real])
            self.unlabeled_idx = self.unlabeled_idx[unlabeled]
            if len(real) == 0 :
                return FINISH

        if len(self.unlabeled_idx) == 0 :
            return FINISH
        else :
            return RUN

