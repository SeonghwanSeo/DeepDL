import torch
import torch.nn as nn
import types

class DefaultModel(nn.Module) :
    #Static attribute
    default_parameter = {}
    loss_fn = None

    def __init__(self):
        super().__init__()

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
            self.load_state_dict(torch.load(load_save_file)) 
        else:
            for param in self.parameters():
                if param.dim() == 1:
                    continue
                else:
                    nn.init.xavier_normal_(param)
        
        self.device = device
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(self)
            model.device = device
        else :
            model = self

        model.to(device)
        return self.trainer(model)

# Add method
def default_trainer(model) :
    def train_step(self, sample, optimizer, gradient_clip_val = 0.0) :
        _, loss = self._step(sample)
        loss.backward()
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        optimizer.step()
        self.zero_grad()
        return loss

    @torch.no_grad()
    def val_step(self, sample) :
        _, loss = self._step(sample)
        return loss
    
    def save_model(self, save_file) :
        """
        save_model()
        save_model() works the same as function torch.save().
        Due to torch.save() save the Paralleled model(nn.DataParallel) when we trained the model in multi-gpu environment, we define save_model().
        """

        if torch.cuda.device_count() > 1:
            torch.save(self.module.state_dict(), save_file)
        else:
           torch.save(self.state_dict(), save_file)

    model.train_step = types.MethodType(train_step, model)
    model.val_step = types.MethodType(val_step, model)
    model.save_model = types.MethodType(save_model, model)
    return model
