# process.py -- this is file to read img from data:food11
# 2025/8/2
# Le Jiang

import torch
import torch.utils as utils
import torchvision.transforms.v2 as v2  
from PIL import Image
import os
import doctest
import numpy as np
import random

# data pre-procession
def read_images(imgs_dir_root, data_tag='training'):
    '''
    ---
    PARAS
    ---
    imgs_path: str, the file storaged the images
    data_tag: str, 'training', 'test', 'validation'

    ---
    RETURN
    --- 
    return a torch array that storages the images data

    ---
    TESTS
    ---
    >>> images_list, labels_list, nimages_list = read_images('./data/food11', 'training')
    >>> print(len(images_list))
    9866
    >>> print(len(images_list) == len(labels_list))
    True
    >>> print(type(labels_list[0]))
    <class 'int'>
    >>> print(nimages_list[994])
    10_0.jpg
    >>> print(labels_list[994])
    10

    '''
    imgs_dir_root = os.path.join(imgs_dir_root, data_tag)
    
    images_list, labels_list, nimages_list = [], [], []
    nimages_list = os.listdir(imgs_dir_root)
    
    if data_tag in ['training', 'validation']:
        labels_list = [int(nimage.split('_')[0]) for nimage in nimages_list]
        
    elif data_tag in ['test']:
         labels_list = [-1] * len(nimages_list)

    # Process images with explicit file closing
    for nfile in nimages_list:
        file_path = os.path.join(imgs_dir_root, nfile)
        with Image.open(file_path) as img:
            img = img.convert('RGB')  # Ensure RGB format
            images_list.append(img.copy())  # Create a copy to detach from underlying file
    
    return images_list, labels_list, nimages_list
      

if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)