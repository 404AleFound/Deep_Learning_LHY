# train.py -- 
# Le Jiang
# 2025/8/25

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import math
import os
from models.conv import Autoencoder_Conv2d
from models.fcn import Autoencoder_Fcn
# from models.vae import 
from dataset.dataset import My_Dataset
from train import train
from utils.logger import create_filelogger
from utils.seed import all_seed
from datetime import datetime

def train(train_data_loader, model, device, config, logger_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    best_loss = math.inf
    for epoch in range(config['n_epoch']):
        # train the model
        model.train()
        trian_loss_list = []
        train_loss_perepoch = 0.0
        step = 1
        for imgs in tqdm(train_data_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            logger_train.info(f'epoch {epoch+1} step {step} train loss {loss.item()}')
            trian_loss_list.append(loss.item())
        train_loss_perepoch = sum(trian_loss_list)/len(trian_loss_list)
        logger_train.info(f'EPOCH {epoch+1} train loss {train_loss_perepoch}')
        print(f'EPOCH {epoch+1} train loss {train_loss_perepoch}')

        # save the checkpoints
        if train_loss_perepoch < best_loss:
            best_loss = train_loss_perepoch
            torch.save(model, f'{config['checkpoints_dir']}/best_model.pt')
            print(f'model with best loss saved.')
        if (epoch+1) % 1 == 0:
            torch.save(model, f'{config['checkpoints_dir']}/epoch_{epoch+1}_model.pt')
            print(f'model at epoch {epoch+1} saved.')
        


if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        # ==================================
        'data_path':'./data/ml2022spring-hw8/trainingset.npy',
        'checkpoints_dir':f'./checkpoints/{timestamp}',
        'logger_dir':f'./logger/{timestamp}/train',
        # ==================================
        'batch_size': 1000,
        'num_worker': 0,
        'lr': 0.001,
        'n_epoch':5,
        'seed': 666
    }

    all_seed(config['seed'])

    if not os.path.exists(config['checkpoints_dir']):
        os.makedirs(config['checkpoints_dir'], mode=0o755)

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