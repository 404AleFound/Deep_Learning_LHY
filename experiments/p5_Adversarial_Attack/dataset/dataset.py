# dataset.py -- 
# Le Jiang 
# 2025/9/1

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cifar_10_mean = (0.491, 0.482, 0.447) # cifar_10 图片数据三个通道的均值
cifar_10_std = (0.202, 0.199, 0.201) # cifar_10 图片数据三个通道的标准差

mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir='./data', transform=transform):
        self.images = []
        self.names = []
        self.labels = []
        for i, class_dir in enumerate(sorted(glob(f'{data_dir}/*/'))):
            images = sorted(glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        if transform:
            self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        return image, label
    
    def __getname__(self, index):
        return self.names[index]
        

if __name__ == '__main__':
    mydataset = AdvDataset()
    print(mydataset[4], mydataset.__getname__(4))
    