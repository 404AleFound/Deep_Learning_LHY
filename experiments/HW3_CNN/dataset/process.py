# process.py -- this is file to read img from data:food11
# 2025/8/2
# Le Jiang

import torch.utils as utils
import torchvision.transforms.v2 as v2
import PIL
import os
import doctest

# data pre-procession
def read_images(imgs_dir_root, data_tag='training', resize_shape=128):
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
    >>> print(images_list[0].shape)
    torch.Size([3, 128, 128])
    >>> print(type(labels_list[0]))
    <class 'int'>
    >>> print(nimages_list[5])
    0_102.jpg
    >>> print(labels_list[5])
    0

    '''
    imgs_dir_root = os.path.join(imgs_dir_root, data_tag)
    test_tfm = v2.Compose([
        v2.Resize((resize_shape, resize_shape)),
        v2.ToTensor(),
        v2.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ])

    train_tfm = v2.Compose([
        v2.Resize((resize_shape, resize_shape)),
        # AutoAugment: Learning Augmentation Strategies from Data "<https://arxiv.org/pdf/1805.09501.pdf>"
        # v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
        v2.Resize((128, 128)),
        v2.RandomHorizontalFlip(p=1),   # 随机水平翻转
        v2.RandomVerticalFlip(p=1),     # 随机上下翻转
        v2.RandomGrayscale(0.5), # 随机灰度化
        v2.RandomSolarize(threshold=192.0),
        v2.ColorJitter(brightness=.5,hue=0.5), # 改变图像的亮度和饱和度
        v2.RandomRotation(degrees=(0, 180)), # 图像随机旋转
        v2.RandomInvert(),# 改变图像的颜色
        v2.ToTensor(),
        v2.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ])

    if data_tag == 'training' or data_tag == 'validation':
        nimages_list = os.listdir(imgs_dir_root)
        # need a shuffle to the nimages_list
        labels_list = [int(nimage.split('_')[0]) for nimage in nimages_list]
        images_list = [train_tfm(PIL.Image.open(os.path.join(imgs_dir_root, nfile))) 
                      for nfile in nimages_list]

    elif data_tag == 'test':
        nimages_list = os.listdir(imgs_dir_root)
        images_list = [test_tfm(PIL.Image.open(os.path.join(imgs_dir_root, nfile))) 
                      for nfile in nimages_list]
        labels_list = None
    
    return images_list, labels_list, nimages_list

if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)