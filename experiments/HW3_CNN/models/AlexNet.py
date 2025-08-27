# AlexNet.py -- AlexNet network for images classification
# 2025/8/2
# Le Jiang

import torch.nn as nn
import doctest

class AlexNet(nn.Module):
    '''
    ---
    TESTS
    ---
    >>> import torch
    >>> x = torch.rand(1, 3, 224, 224) # should a batch_size parameter
    >>> alexnet = AlexNet()
    >>> output = alexnet(x)
    >>> print(output.shape)
    torch.Size([1, 11])
    '''
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(96)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(256),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
        )

        self.layer6 = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.layer8 = nn.Sequential(
            nn.Linear(4096, 11),#(batch_size, 11)
            nn.Softmax(dim=1),
        )

    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        output = self.layer8(x7)
        return output

if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)