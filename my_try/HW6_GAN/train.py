# train.py -- train the network
# Le Jiang 
# 2025/8/16

import torch
import torch.nn as nn
from models.gan import Generator, Discriminator
from torch.autograd import Variable
from dataset.dataset import Dataset
from tqdm import tqdm

config = {
    # dataset path
    'datapath':'./data/faces',

    # train parameters
    'n_epoch':300,

    # model checkpoints save path
    'best_checkpoints':f'./checkpoints/{timestamp}/best',
    'latest_checkpoints':f'./checkpoints/{timestamp}/latest',

    # train log save path
    'logs_dir':'./logs/',
    'logs_train':f'./logs/{timestamp}/train',
    'logs_eval':f'./logs/{timestamp}/eval',
}


def train(dataloader, model_g, model_d, config, device):
    criterion = nn.BCELoss()
    optimizm_g = torch.optim.Adam(model_g.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizm_d = torch.optim.Adam(model_d.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    steps = 0

    for epoch in range(config['n_epoch']):
        dataloader_pbar = tqdm(dataloader)
        for imgs in dataloader_pbar:
            imgs = imgs.to(device)
            bs = imgs.size[0]

            z = Variable(torch.randn(bs, config['z_dim'])).to(device)
            f_imgs = model_g(z)
            r_imgs = Variable(imgs).to(device)
            r_labels = torch.ones((bs)).to(device)
            f_labels = torch.zeros((bs)).to(device)

            r_scores = model_d(r_imgs)
            f_scores = model_d(f_imgs)

            # Discriminator's loss setting:
            # GAN: loss_D = (r_loss + f_loss)/2
            # WGAN: loss_D = -torch.mean(real) 
            
            r_loss = criterion(r_scores, r_labels)
            f_loss = criterion(f_scores ,f_labels)
            loss_d = (r_loss + f_loss) / 2

            model_d.zero_grad()
            loss_d.backward()
            optimizm_d.step()

            if steps % config['n_critic'] == 0:
                z = Variable(torch.randn(bs, config['z_dim'])).to(device)
                f_imgs = model_g(z)
                f_scores = model_d(f_imgs)
                loss_g = criterion(f_scores, r_labels)
                model_g.zero_grad()
                loss_g.backward()
                optimizm_g.step()

            setps+=1
        
        model_g.eval()

