import argparse

def parser():
    parser = argparse.ArgumentParser(description='parser for training')
    parser.add_argument('--ngpu',
                        help='number of gpus',
                        type=int,
                        default=0)
    parser.add_argument('--num_workers',
                        help='number of workers(cpu)',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=1)
    parser.add_argument('--epoch',
                        help='training epoch',
                        type=int,
                        default=300)
    parser.add_argument('--save_file',
                        help='path for model save file',
                        type=str,
                        default='./save.pt')
    parser.add_argument('--config',
                        help='config file',
                        type=str,
                        default='config/test.yaml')
    args = parser.parse_args()
    return args
