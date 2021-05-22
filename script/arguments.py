import argparse

def parser(model = 'rnn'):
    config = {'rnn': 'config/rnn.yaml', 'gcn': 'config/gcn.yaml'}
    parser = argparse.ArgumentParser(description='parser for training')
    parser.add_argument('--ngpu',
                        help='number of gpus',
                        type=int,
                        default=1)
    parser.add_argument('--num_workers',
                        help='number of workers(cpu)',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=100)
    parser.add_argument('--epoch',
                        help='training epoch',
                        type=int,
                        default=100)
    parser.add_argument('--save_file',
                        help='path for model save file',
                        type=str,
                        default='./save.pt')
    parser.add_argument('--config',
                        help='config file',
                        type=str,
                        default=config[model])
    args = parser.parse_args()
    return args
