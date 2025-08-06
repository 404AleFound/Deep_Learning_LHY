import torch

config = {
        'seed': 666,
        'data_dir': './data/food11',
        'checkpoint_dir': './checkpoints',
        'num_epoch': 1000,
        'batch_size': 32,
        'learning_rate':0.0003,
        'weight_decay':1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'early_stop': 300,
        'clip_flag': True,
    }