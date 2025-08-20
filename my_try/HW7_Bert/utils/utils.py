# utils.py -- storage some utils function
# Le Jiang
# 2025/8/20

import logging
import numpy as np
import torch
import os
import random

def create_logger(logger_name, logger_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    hander = logging.FileHandler(logger_path)
    hander.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(hander)
    logger.propagate = False
    return logger

def all_seed(seed = 6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed) 
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed) 
    # python 全局
    os.environ['PYTHONHASHSEED'] = str(seed) 
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')