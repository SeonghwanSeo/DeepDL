import pickle
import numpy as np
import torch
import os
import argparse
from models.RNN import RNN
from models.Transformer import TransformerModel
import utils
from rdkit import Chem
from rdkit.Chem.QED import qed
from sklearn.metrics import roc_curve, roc_auc_score
import time
import yaml
from quantifier import quantifier

parser = argparse.ArgumentParser()
parser.add_argument('--config', default = 'config/test_worlddrug.yaml')

args = parser.parse_args()
print('config :', args.config)
print()

with open(args.config, 'r') as f :
    config = yaml.safe_load(f)
    data_config = config['data_config']
    model_params = config['model_params']
    model_params['dropout']=0.0

ngpu=1
cmd = utils.set_cuda_visible_device(ngpu)
os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c_to_i = pickle.load(open(data_config['c_to_i'], 'rb'))
i_to_c = pickle.load(open(data_config['i_to_c'], 'rb'))
n_char = len(c_to_i)

#=========== model =============
if model_params['kind']=='RNN' :
    model = RNN(model_params, n_char, i_to_c)
elif model_params['kind']=='Transformer':
    model = TransformerModel(model_params, n_char, i_to_c)
else :
    print("ERR : Not allowed model. Default model is RNN")
    exit(1)

model = utils.initialize_model(model, device, data_config['save_file'])
model.eval()

print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print()
mod_quantifier = quantifier(model, c_to_i, model_params['stereo'])
#============= load testset data =============
data_smiles = {}

for tf, data_list in enumerate([data_config['negative_set'], data_config['positive_set']]) : #negative : 0, positive : 1
    for data in data_list :
        with open(data) as f :
            smiles_list = f.readlines()
            smiles_list = [s.strip().split('\t')[1] for s in smiles_list]
        data_smiles[data] = (tf, smiles_list)

#============= quantifying drug-likeness  ================
data_result = {}
with torch.no_grad() :
    for data in data_smiles.keys() :
        tf, smiles_list = data_smiles[data]
        answer = []
        qed_result = []
        mod_result = []
        for smiles in smiles_list : 
            answer.append(tf)
            mod_result.append(mod_quantifier.mol_to_score(smiles)) # To be modified
            mol = Chem.MolFromSmiles(smiles)
            qed_value = qed(mol)
            qed_result.append(qed(mol))
        data_result[data] = (answer, mod_result, qed_result)

#============= AUROC scoring of model ================
for pos_set in data_config['positive_set'] :
    for neg_set in data_config['negative_set'] :
        pos_name = pos_set.split('/')[-1]
        neg_name = neg_set.split('/')[-1]

        pos_result = data_result[pos_set]
        neg_result = data_result[neg_set]

        print("===============================")
        print(f"positive set : {pos_name:<10} ({len(pos_result[0])} smiles)")
        print(f"negative set : {neg_name:<10} ({len(neg_result[0])} smiles)")
        print()

        answer = np.array(pos_result[0] + neg_result[0])
        mod_score = -np.array(pos_result[1] + neg_result[1])
        qed_score = np.array(pos_result[2] + neg_result[2])
        
        mod_score = roc_auc_score(answer, mod_score)
        qed_score = roc_auc_score(answer, qed_score)
        
        print(f"model  {mod_score:.4f}")
        print(f"qed    {qed_score:.4f}")
        print(f"data : {data_config['save_file'][:-8]}_{pos_name}_{neg_name}.pkl\n")
        with open(str(f"{data_config['save_file'][:-8]}_{pos_name}_{neg_name}.pkl"), 'wb') as w :
            pickle.dump((answer,qed_score,mod_score), w)
