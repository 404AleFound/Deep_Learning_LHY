# main.py --
# Le Jiang
# 2025/8/25

import torch
from torch.utils.data import DataLoader
import os
from models.conv import Autoencoder_Conv2d
from models.fcn import Autoencoder_Fcn
# from models.vae import 
from dataset.dataset import My_Dataset
from train import train
from utils.logger import create_filelogger
from utils.seed import all_seed
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

config = {
    # ==================================
    'data_path':'./data/ml2022spring-hw8/trainingset.npy',
    'checkpoints_dir':f'./checkpoints/{timestamp}',
    'logger_dir':f'./logger/{timestamp}/train',
    'logger_dir':f'./logger/{timestamp}/note',
    # ==================================
    'batch_size': 1000,
    'num_worker': 0,
    'lr': 0.001,
    'n_epoch':5,
    'seed': 666
}



all_seed(config['seed'])

if not os.path.exists(config['checkpoints_dir']):
    os.makedirs(config['checkpoints_dir'])

if not os.path.exists(config['logger_dir']):
    os.makedirs(config['logger_dir'], mode=0o755)

logger_train = create_filelogger('train_logger', config['logger_dir'])

my_dataset = My_Dataset(config['data_path'])

train_data_loader = DataLoader(
    my_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_worker'],
    pin_memory=True,
    drop_last=True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Autoencoder_Conv2d().to(device)

train(train_data_loader, model, device, config, logger_train)
