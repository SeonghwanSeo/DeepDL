import random
import numpy as np

def load_rnn_data(data_config):
    #model: both model class and model object
    with open(data_config['positive_file']) as f : 
        train_data = f.readlines()
        train_data = [s.strip().split('\t')[-1] for s in train_data]
    return np.array(train_data)

def load_gcn_data(data_config):
    with open(data_config['positive_file']) as f :
        pos_file = f.readlines()
        pos_file = [l.strip().split('\t')[-1] for l in pos_file]
    with open(data_config['negative_file']) as f :
        neg_file = f.readlines()
        neg_file = [l.strip().split('\t')[-1] for l in neg_file]
    
    train_data = pos_file + neg_file
    answer = [1 for _ in range(len(pos_file))] + [0 for _ in range(len(neg_file))]
        
    return np.array(train_data), np.array(answer)

"""
def load_rdkit_data(data_config) :
    f = np.load(data_config['positive_file'])
    pos_data = f['feature']
    f.close()
    f = np.load(data_config['negative_file'])
    neg_data = f['feature']
    f.close()
    
    train_data = np.append(pos_data, neg_data)
    answer = [1] * len(pos_data) + [0] * len(neg_data)
    return train_data, np.array(answer)
"""     
