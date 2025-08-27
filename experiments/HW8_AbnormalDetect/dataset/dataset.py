# dataset.py -- dataset for hw7 abnormaldetect
# Le Jiang
# 2025/8/24

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt 
import doctest

class My_Dataset(Dataset):
    '''
    ---
    TEST
    ---
    >>> my_dataset = My_Dataset('data/ml2022spring-hw8/trainingset.npy')
    >>> print(len(my_dataset))
    100000
    >>> print(my_dataset[0].max().item())
    0.9843137264251709
    >>> print(my_dataset[0].shape)
    torch.Size([3, 64, 64])
    >>> plt.imshow(my_dataset[0].permute(1, 2, 0))
    <matplotlib.image.AxesImage object at 0x000001CECDD40440>
    >>> plt.show()
    '''
    def __init__(self, data_path):
        self.imgs = np.load(data_path)# (100000, 64, 64, 3)
        if self.imgs.shape[-1] == 3:
            np.permute_dims(self.imgs, [0, 3, 1, 2])# (100000, 3, 64, 64)
        self.procession = transforms.Compose([
            # in range(0,255) -> in range(0,1)
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32)),
        ])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.procession(img)
        return img
    
if __name__ == '__main__':
    doctest.testmod(verbose=True)