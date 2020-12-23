from .transformer import TransformerModel
from .rnn import RNN
from .gcn import GCNModel
from .bayesian_gcn import BayesianGCNModel
_model_list = [RNN, TransformerModel, GCNModel, BayesianGCNModel]

MODEL_LIST = {model.__name__:model for model in _model_list}
