import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import types
from omegaconf import OmegaConf
import os

def construct_balanced_sampler(answer) :
    n_total = len(answer)
    n_true = int(sum(answer))
    n_false = n_total - n_true
    n_sample = 2 * min(n_true, n_false)
    weights = np.where(answer == 1, 1/n_true, 1/n_false)
    return WeightedRandomSampler(weights, n_sample)

class DefaultModel(nn.Module) :
    #Static attribute
    default_parameter = {}
    loss_fn = None

    def __init__(self):
        super().__init__()

    @property
    def device(self) :
        return torch.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")

    @classmethod
    def initialize_hyperparams(cls, model_params) :
        #======== Set default value for omitted parameters=======#
        for key, value in cls.default_parameters.items() :
            if key not in model_params :
                model_params[key] = value
    
    def initialize_model(self, device, load_save_file=None):
        """
        To set device of model
        If there is save_file for model, load it
        """
        if load_save_file:
            self.load_state_dict(torch.load(load_save_file, map_location=device))
        else:
            for param in self.parameters():
                if param.dim() == 1:
                    continue
                else:
                    nn.init.xavier_normal_(param)
        
        if hasattr(self, '_training_step') : return self

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(self)
            model.device = device
        else :
            model = self

        model.to(device)
        return self.trainer(model)

    def test(self, smiles: str):
        pass

    def construct_dataset(self, data) :
        pass

    def construct_dataloader(self, train_data, val_data, batch_size, num_workers, sampler = None) :
        if train_data is not None :
            train_dataset = self.construct_dataset(train_data)
            collate_fn = getattr(train_dataset, 'collate_fn', None)
            if sampler == 'balanced' :
                sampler = construct_balanced_sampler(train_data[1])
                train_dataloader = DataLoader(train_dataset, batch_size, num_workers=num_workers, collate_fn=collate_fn, sampler = sampler)
            else :
                train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, sampler = sampler)
        else :
            train_dataloader = None

        if val_data is not None :
            val_dataset = self.construct_dataset(val_data)
            collate_fn = getattr(val_dataset, 'collate_fn', None)
            val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        else :
            val_dataloader = None

        return train_dataloader, val_dataloader

    @classmethod
    def load_model(cls, model_path, map_location = 'cpu') :
        parameter_path = os.path.join(model_path, 'save.pt')
        config_path = os.path.join(model_path, 'config.yaml')
        config = OmegaConf.load(config_path)
        model_params = config.model
        model = cls(model_params)
        model.initialize_model(map_location, parameter_path)
        return model

# Add method
def default_trainer(model) :
    def _training_step(self, sample, optimizer, gradient_clip_val) :
        _, loss = self._step(sample)
        loss.backward()
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        optimizer.step()
        self.zero_grad()
        return loss

    @torch.no_grad()
    def _validation_step(self, sample) :
        _, loss = self._step(sample)
        return loss

    def train_epoch(self, train_dataloader, val_dataloader, optimizer, gradient_clip_val = 0.0) :
        self.train()
        loss_list = []
        for sample in train_dataloader :
            loss = self._training_step(sample, optimizer, gradient_clip_val) 
            loss_list.append(loss.data.cpu().numpy())
        train_loss = np.mean(np.array(loss_list))
        if val_dataloader is not None :
            self.eval()
            loss_list = []
            for sample in val_dataloader :
                loss = self._validation_step(sample) 
                loss_list.append(loss.data.cpu().numpy())
            val_loss = np.mean(np.array(loss_list))
        else :
            val_loss = None

        return train_loss, val_loss

    @torch.no_grad()
    def test_samples(self, test_dataloader) :
        self.eval()
        y_pred_list = []
        for sample in test_dataloader :
            y_pred, _ = self._step(sample)
            y_pred_list.append(y_pred)
        return torch.cat(y_pred_list, dim=0).cpu().numpy()
    
    def save_model(self, save_file) :
        """
        save_model() works the same as function torch.save().
        Due to torch.save() save the Paralleled model(nn.DataParallel) when we trained the model in multi-gpu environment,
        we define save_model().
        """
        if save_file is not None :
            if torch.cuda.device_count() > 1:
                torch.save(self.module.state_dict(), save_file)
            else:
               torch.save(self.state_dict(), save_file)

    model._training_step = types.MethodType(_training_step, model)
    model._validation_step = types.MethodType(_validation_step, model)
    model.train_epoch = types.MethodType(train_epoch, model)
    model.save_model = types.MethodType(save_model, model)
    model.test_samples = types.MethodType(test_samples, model)
    return model
