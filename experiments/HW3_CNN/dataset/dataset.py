# food11.py -- load the food11 dataset
# 2025/8/1
# Le Jiang

import torch.utils as utils
import doctest

# the dir structure of "./ml2022spring-hw3b":
# data
# |__ food11
#       |__ test: images for model testing
#       |__ training: images for model training
#       |__ validation: images for model validation


class Food_Dataset(utils.data.Dataset):
    def __init__(self, images_list, labels_list=None):
        super(Food_Dataset, self).__init__()
        self.images = images_list
        self.labels = labels_list

    def __len__(self):
        if self.labels is not None:
            return (len(self.images) == len(self.labels)) * len(self.images)
        return len(self.images)
    
    def __getitem__(self, index):
        if self.labels is not None:
            return self.images[index], self.labels[index]
        return self.images[index]

if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)