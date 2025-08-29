# ResNet.py -- 
# Le Jiang
# 2025/8/26

import torch
import torch.nn as nn
import doctest

# out_size = (in_size - kernal_size + 2 * padding)/step + 1
class CNN_Blcok(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_features):
        super(CNN_Blcok, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features),
            )
    
    def forward(self, x):
        x = self.block(x)
        return x
    
class ResNet(nn.Module):
    '''
    ---
    TESTS
    ---
    >>> x = torch.rand(1, 3, 128, 128) # should a batch_size parameter
    >>> resnet = ResNet()
    >>> output = resnet(x)
    >>> print(output.shape)
    torch.Size([1, 11])
    '''
    def __init__(self):
        super(ResNet, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            CNN_Blcok(3, 64, 3, 2, 1, 64))
        
        self.cnn_layer2 = nn.Sequential(
            CNN_Blcok(64, 64, 3, 1, 1, 64))
        
        self.cnn_layer3 = nn.Sequential(
            CNN_Blcok(64, 128, 3, 2, 1, 128))
        
        self.cnn_layer4 = nn.Sequential(
            CNN_Blcok(128, 128, 3, 1, 1, 128))
        
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 32 * 32, 128 * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 128 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 11))

        self.relu = nn.Sequential(nn.ReLU())

        self.pooling = nn.Sequential(nn.MaxPool2d(2, 2, 0))

    # out_size = (in_size - kernal_size + 2 * padding)/step + 1
    def forward(self, x):# 3, 128, 128
        x1 = self.cnn_layer1(x)# 64, 64, 64
        x1 = self.relu(x1)# 64, 64, 64
        x2 = self.cnn_layer2(x1)# 64, 64, 64
        x2 = x1 + x2# 64, 64, 64
        x2 = self.relu(x2)# 64, 64, 64

        x3 = self.cnn_layer3(x2)# 128, 32, 32
        x3 = self.relu(x3)# 128, 32, 32
        x4 = self.cnn_layer4(x3)# 128, 32, 32
        x4 = x3 + x4# 128, 32, 32
        x4 = self.relu(x4)# 128, 32, 32

        xout = x4.flatten(1)# 128 * 32 * 32
        xout = self.fc_layer(xout)
        return xout
    
if __name__ =='__main__':
    doctest.testmod(verbose=True)