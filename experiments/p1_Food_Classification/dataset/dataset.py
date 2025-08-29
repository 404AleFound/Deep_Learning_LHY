# food11.py -- load the food11 dataset
# 2025/8/1
# Le Jiang

import torch.utils as utils
import doctest
import torch
import torchvision
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt  
import random

import numpy as np
# the dir structure of "./ml2022spring-hw3b":
# data
# |__ food11
#       |__ test: images for model testing
#       |__ training: images for model training
#       |__ validation: images for model validation

def img_enhancement(img, tag='training', reshape_size=(128,128)):
    if tag in ['training']:
        trm = torchvision.transforms.Compose([
            v2.Resize(reshape_size),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            torchvision.transforms.RandomResizedCrop(reshape_size, scale=(0.5, 1), ratio=(0.5, 2)),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
        ])
        return trm(img)
    if tag in ['test', 'validation']:
        trm = torchvision.transforms.Compose([
            v2.Resize(reshape_size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
        ])
        return trm(img)

class Food_Dataset(utils.data.Dataset):
    '''
    ---
    TEST
    ---
    >>> from process import read_images
    >>> train_images_list, train_labels_list, nimages_list = read_images('./data/food11/', 'training')
    >>> train_set = Food_Dataset(train_images_list, train_labels_list, nimages_list, 'training', (128,128))
    >>> x, y, _ = train_set[994]
    >>> print(y)
    10
    >>> x, y, _ = train_set[1200]
    >>> print(y)
    10
    '''
    def __init__(self, imgs_list, labels_list, nimages_list, tag='training', reshape_size=(128,128)):
        super(Food_Dataset, self).__init__()
        self.images = imgs_list
        self.labels = labels_list
        self.nimages = nimages_list
        self.tag = tag
        self.reshape_size = reshape_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = img_enhancement(self.images[index], self.tag, self.reshape_size)
        return img, self.labels[index], self.nimages[index]
    

def show(reshape_size=(128,128), tag='training'):
    from process import read_images
    train_images_list, train_labels_list, nimages_list = read_images('./data/food11/', tag)

    train_set = Food_Dataset(train_images_list, train_labels_list, nimages_list, tag, reshape_size)

    del train_images_list, train_labels_list, nimages_list
    
    selected_indices = random.sample(range(len(train_set)), min(25, len(train_set)))
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    plt.suptitle(f"from total {len(train_set)} imgs randomly select 25 imgs", fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(selected_indices):
        row = i // 5
        col = i % 5
        img, label, filename = train_set[idx]  
        
        if img is not None:
            img = img.cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

        axes[row, col].imshow(img)
        
        title = f"Class: {label}\n{filename}"
        
        axes[row, col].set_title(title, fontsize=9, pad=5)
        axes[row, col].axis('off')
    
    for i in range(len(selected_indices), 25):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # doct = doctest.testmod(verbose=True)
    show()