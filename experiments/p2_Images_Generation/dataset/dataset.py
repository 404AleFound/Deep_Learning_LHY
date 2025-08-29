# dataset.py -- turn the .jpg files into torch and pack into torch.dataset
# Le Jiang
# 2025/8/10
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os 
import doctest

class My_Dataset(Dataset):
    '''
    ---
    TEST
    ---
    >>> dataset = My_Dataset('./data/faces')
    >>> print(dataset[0].shape)
    torch.Size([3, 64, 64])
    '''
    def __init__(self, data_dir):
        super(My_Dataset, self).__init__()
        self.data_dir = data_dir
        self.img_names = os.listdir(data_dir)
        self.transforms_pipline = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        fname_img = self.img_names[index]
        img_pil = Image.open(os.path.join(self.data_dir, fname_img))
        return self.transforms_pipline(img_pil)
    
    def __len__(self):
        return len(self.img_names)

if __name__ =='__main__':
    doct = doctest.testmod(verbose=True)