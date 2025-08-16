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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    # dataset path
    'datapath':'./data/faces',

    # train parameters
    'n_epoch':10,
    'z_dim':100,
    'lr':0.0001,
    'n_critic': 1,
    # model checkpoints save path
    'best_checkpoints':f'./checkpoints/{timestamp}/best',
    'latest_checkpoints':f'./checkpoints/{timestamp}/latest',

    # train log save path
    'log_dir':f'./logs/{timestamp}',
    'log_g':f'./logs/{timestamp}/g.txt',
    'log_d':f'./logs/{timestamp}/d.txt',
}

if not os.path.exists(config['log_dir']):
    os.makedirs(config['log_dir'])
# if not os.path.exists(config['log_d']):
#     os.makedirs(config['log_d'])

g_logger = logging.getLogger('g')
g_logger.setLevel(logging.INFO)
g_hander = logging.FileHandler(config['log_g'])
g_hander.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
g_logger.addHandler(g_hander)
g_logger.propagate = False

d_logger = logging.getLogger('d')
d_logger.setLevel(logging.INFO)
d_hander = logging.FileHandler(config['log_d'])
d_hander.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
d_logger.addHandler(d_hander)
d_logger.propagate = False

def train(dataloader, model_g, model_d, config, device):
    criterion = nn.BCELoss()
    optimizm_g = torch.optim.Adam(model_g.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizm_d = torch.optim.Adam(model_d.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    steps = 0

    for epoch in range(config['n_epoch']):
        dataloader_pbar = tqdm(dataloader)
        for imgs in dataloader_pbar:
            # initially get the images
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # train the discriminator
            model_d.train()
            z = Variable(torch.randn(bs, config['z_dim'])).to(device)
            f_imgs = model_g(z)
            r_imgs = Variable(imgs).to(device)
            r_labels = torch.ones((bs)).to(device)
            f_labels = torch.zeros((bs)).to(device)
            r_scores = model_d(r_imgs)
            f_scores = model_d(f_imgs)
            r_loss = criterion(r_scores, r_labels)
            f_loss = criterion(f_scores ,f_labels)
            loss_d = (r_loss + f_loss) / 2
            d_logger.info(f"Step {steps} - D_Loss: {loss_d:.4f}")
            model_d.zero_grad()
            loss_d.backward()
            optimizm_d.step()

            # train the geneator
            model_g.train()
            if steps % config['n_critic'] == 0:
                z = Variable(torch.randn(bs, config['z_dim'])).to(device)
                f_imgs = model_g(z)
                f_scores = model_d(f_imgs)
                loss_g = criterion(f_scores, r_labels)
                g_logger.info(f'Step {steps} - G_Loss: {loss_g:.4f}')
                model_g.zero_grad()
                loss_g.backward()
                optimizm_g.step()

            steps+=1
        
        model_g.eval()
        input = torch.randn(100, config['z_dim']).to(device)
        output = (model_g(input).data+1)/2.0
        filename = os.path.join(config['log_dir'], f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(output, filename, nrow=10)
        logging.info(f'Save some samples to {filename}.')

if __name__ == '__main__':
        
    from dataset.dataset import My_Dataset
    from models.gan import Generator, Discriminator
    from torch.utils.data import DataLoader
    dataset = My_Dataset('./data/faces')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    generator = Generator(config['z_dim']).to(device)
    discriminator = Discriminator(3).to(device)
    train(dataloader, generator, discriminator, config, device)