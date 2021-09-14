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
    
    if data_config.get('balance', False) == True :
        if len(pos_file) < len(neg_file) :
            random.shuffle(neg_file)
            neg_file = neg_file[:len(pos_file)]
        elif len(pos_file) > len(neg_file) :
            random.shuffle(pos_file)
            pos_file = pos_file[:len(neg_file)]
    
    train_data = pos_file + neg_file
    answer = [1 for _ in range(len(pos_file))] + [0 for _ in range(len(neg_file))]
        
    return np.array(train_data), np.array(answer)
