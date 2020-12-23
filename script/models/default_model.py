from torch import sigmoid
import torch.nn as nn
from .cdropout import ConcreteDropout as CDr

class DefaultModel(nn.Module) :
    #Static attribute
    modeltype = 'PytorchModel'
    require_datatype = None
    default_parameter = {}
    default_train_parameter = {}
    default_parameter_keys = []
    default_train_parameter_keys=[]

    def __init__(self):
        super().__init__()
    
    def regularisation(self):
        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            total_regularisation += module.rr
        return total_regularisation
    
    def get_p(self):
        p = []
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            p.append(float(sigmoid(module.p_logit)))
        return p
    
    def init_cdropout(self):
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            module.__init__(module.weight_regularizer, module.dropout_regularizer)

    @classmethod
    def initialize_params(cls, model_params) :
        #======== Set default value for omitted parameters=======#
        model_params['model'] = cls.__name__
        for key, value in cls.default_parameters.items() :
            if key not in model_params :
                model_params[key] = value
        if getattr(cls, "default_parameters_keys", None) is None :
            keys = list(cls.default_parameters.keys())
            keys.sort()
        else :
            keys = cls.default_parameters_keys
        return ['model'] + keys

    @classmethod 
    def initialize_train_params(cls, train_params) :   
        for key, value in cls.default_train_parameters.items() :
            if key not in train_params.keys() :
                train_params[key] = value
       
        #Set initial model(pretrained model) with default value None
        if 'init_model' not in train_params.keys() :
            train_params['init_model'] = None

        #Set Validation Mode with default mode 'val'. ('val'/'5cv')
        if 'valid_mode' not in train_params.keys() :
            train_params['valid_mode'] = 'val'
        
        keys = list(cls.default_train_parameters.keys())
        keys.sort()
        return ['init_model', 'valid_mode'] + keys
   
