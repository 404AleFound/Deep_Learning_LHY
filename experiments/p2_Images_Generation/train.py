# train.py -- train the network
# Le Jiang 
# 2025/8/16

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime
import torchvision
import logging
import os
from utils import all_seed, create_logger, makedirs_
from dataset.dataset import My_Dataset
from models.gan import Generator, Discriminator
from torch.utils.data import DataLoader
from torchsummary import summary
import pprint

def train(dataloader, model_g, model_d, config, device, g_logger, d_logger):
    criterion = nn.BCELoss()
    optimizm_g = torch.optim.Adam(model_g.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizm_d = torch.optim.Adam(model_d.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    steps = 0

    for epoch in range(config['n_epoch']):

        model_d.train()
        model_g.train()
        for imgs in tqdm(dataloader):
            # initially get the images
            imgs = imgs.to(device)
            # train the discriminator
            z = Variable(torch.randn(config['batch_size'], config['z_dim'])).to(device)
            f_imgs = model_g(z)
            r_imgs = Variable(imgs).to(device)
            r_labels = torch.ones((config['batch_size'])).to(device)
            f_labels = torch.zeros((config['batch_size'])).to(device)
            r_scores = model_d(r_imgs)
            f_scores = model_d(f_imgs)
    
            r_loss = criterion(r_scores, r_labels)
            f_loss = criterion(f_scores ,f_labels)
            loss_d = (r_loss + f_loss) / 2
            d_logger.info(f"epoch {epoch} step {steps} d_loss: {loss_d:.4f}")
            model_d.zero_grad()
            loss_d.backward()
            optimizm_d.step()

            # train the geneator
            if steps % config['n_critic'] == 0:
                z = Variable(torch.randn(config['batch_size'], config['z_dim'])).to(device)
                f_imgs = model_g(z)
                f_scores = model_d(f_imgs)
                loss_g = criterion(f_scores, r_labels)
                g_logger.info(f'epoch {epoch} step {steps} - g_loss: {loss_g:.4f}')
                model_g.zero_grad()
                loss_g.backward()
                optimizm_g.step()

            steps+=1
        
        # eval print out imgs
        model_g.eval()
        input = torch.randn(100, config['z_dim']).to(device)
        output = (model_g(input).data+1)/2.0
        filename = os.path.join(config['loggers_dir'], f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(output, filename, nrow=10)
        print(f'Save some samples to {filename}.')

        # save model weights
        torch.save(model_d.state_dict(), os.path.join(config['checkpoints_dir'], f'g_{epoch+1:03d}.pth'))
        torch.save(model_g.state_dict(), os.path.join(config['checkpoints_dir'], f'd_{epoch+1:03d}.pth'))

if __name__ == '__main__':
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        # =================== data dir ===================
        'datapath':'./data/faces',  
        # ------------------------------------------------

        # ================= recorder dir =================
        'recorders_dir':f'./recorders/{timestamp}/',
        'checkpoints_dir':f'./recorders/{timestamp}/checkpoints/',
        'loggers_dir':f'./recorders/{timestamp}/loggers/',
        # ------------------------------------------------

        # ================ hyperparameter ================
        'seeds':6666,
        'n_epoch':50,
        'z_dim':100,
        'lr':0.0001,
        'n_critic': 1,
        'continue':'',
        'batch_size':1024,
        # ------------------------------------------------
    }

    all_seed(config['seeds'])

    makedirs_((config['recorders_dir'], config['checkpoints_dir'], config['loggers_dir']))

    g_logger = create_logger('g', os.path.join(config['loggers_dir'], 'g.log'), show_time=False)
    d_logger = create_logger('d', os.path.join(config['loggers_dir'], 'd.log'), show_time=False)
    note_logger = create_logger('note', os.path.join(config['loggers_dir'], 'note.log'), show_time=True)

    dataset = My_Dataset('./data/faces')
    dataloader = DataLoader(dataset, config['batch_size'], shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    generator = Generator(config['z_dim']).to(device)
    discriminator = Discriminator(3, 64).to(device)

    note_logger.info(f"NOTE: \n[USE] Generator and Discriminator")
    note_logger.info(f"DEVICE:\n{device}")
    note_logger.info(f"GMODEL INFO:\n{str(generator)}")
    note_logger.info(f"GMODEL FORWARD INFO:\nInput:(1, 100)\n{str(summary(generator, (100,)))}")
    note_logger.info(f"DMODEL INFO:\n{str(discriminator)}")
    note_logger.info(f"DMODEL FORWARD INFO:\nInput:(1, 3, 64, 64)\n{str(summary(discriminator, (3, 64, 64)))}")
    note_logger.info(f"CONFIG INFO:\n{pprint.pformat(config, indent=4)}")
    note_logger.info(f"NUM STEPS PER EPOCH (TRAIN):\n{len(dataloader)}")

    train(dataloader, generator, discriminator, config, device, g_logger, d_logger)