# main.py -- start project
# 2025/8/2
# Le Jiang

import torch
import torch.utils as utils
import os

from dataset.process import read_images
from dataset.dataset import Food_Dataset
from models import Residual_Network, Classifier
from models import AlexNet
from utils.seeds import same_seeds, all_seed
from train import train
from config import config

if not os.path.exists(config['checkpoint_dir']):
    os.makedirs(config['checkpoint_dir'])
    
same_seeds(config['seed'])
train_images_list, train_labels_list, _ = read_images(config['data_dir'], 'training')
val_images_list, val_labels_list, _ = read_images(config['data_dir'], 'validation')
test_images_list, _, _ = read_images(config['data_dir'], 'test')
train_set = Food_Dataset(train_images_list, train_labels_list)
val_set = Food_Dataset(val_images_list, val_labels_list)
test_set = Food_Dataset(test_images_list)

del train_images_list, train_labels_list, val_images_list, val_labels_list, test_images_list

train_dataloader = utils.data.DataLoader(train_set, batch_size=config['batch_size'], 
                                         shuffle=True, num_workers=config['num_workers'], 
                                         pin_memory=True)
val_dataloader = utils.data.DataLoader(val_set, batch_size=config['batch_size'], 
                                         shuffle=True, num_workers=config['num_workers'], 
                                         pin_memory=True)

test_dataloader = utils.data.DataLoader(test_set, batch_size=config['batch_size'], 
                                         shuffle=True, num_workers=config['num_workers'], 
                                         pin_memory=True)

print(f"DEVICE:{config['device']}")

my_model = Classifier().to(config['device'])

if os.path.exists(os.path.join(config['checkpoint_dir'], 'latest_model_path')):
    my_model.load_state_dict(torch.load(config['checkpoint_dir'], map_location=config['device']))

train(train_set, val_set, train_dataloader, val_dataloader, config, my_model, config['device'])