# test.py -- 
# Le Jiang 
# 2025/8/27

# import some needed libs
import torch
import torch.utils as utils
import torch.nn as nn
import gc
import os
from tqdm import tqdm
import os
from dataset.process import read_images
from dataset.dataset import Food_Dataset
from models import Residual_Network, BaseNet, AlexNet
from utils.seeds import same_seeds, all_seed
from utils.logger import create_logger
from datetime import datetime
import pprint
import numpy as np

timestamp = '20250827_102518'

config = {
    # =============================================
    'data_dir':'./data/food11',
    'checkpoints_dir':f'./checkpoints/{timestamp}',
    'loggers_dir':f'./loggers/{timestamp}',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_model = BaseNet().to(device)
my_model.load_state_dict(torch.load(f'{config['checkpoints_dir']}/latest_checkpoint.pt',
                                             map_location=device))

test_images_list, test_labels_list, _ = read_images(config['data_dir'], 'test', resize_shape=128)

test_set = Food_Dataset(test_images_list, test_labels_list)

del test_images_list, test_labels_list

test_dataloader = utils.data.DataLoader(test_set, batch_size=config['batch_size'], 
                                        shuffle=True, num_workers=config['num_workers'], 
                                        pin_memory=True, drop_last=True)


my_model.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_dataloader:
        test_pred = my_model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()