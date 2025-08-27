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
    
class Residual_Network(nn.Module):
    '''
    ---
    TESTS
    ---
    >>> x = torch.rand(1, 3, 128, 128) # should a batch_size parameter
    >>> residual_network = Residual_Network()
    >>> output = residual_network(x)
    >>> print(output.shape)
    torch.Size([1, 11])
    '''
    def __init__(self):
        super(Residual_Network, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            CNN_Blcok(3, 64, 3, 1, 1, 64))
        
        self.cnn_layer2 = nn.Sequential(
            CNN_Blcok(64, 64, 3, 1, 1, 64))
        
        self.cnn_layer3 = nn.Sequential(
            CNN_Blcok(64, 128, 3, 2, 1, 128))
        
        self.cnn_layer4 = nn.Sequential(
            CNN_Blcok(128, 128, 3, 1, 1, 128))
        
        self.cnn_layer5 = nn.Sequential(
            CNN_Blcok(128, 256, 3, 2, 1, 256))
        
        self.cnn_layer6 = nn.Sequential(
            CNN_Blcok(256, 256, 3, 1, 1, 256))
        
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11))

        self.relu = nn.ReLU()

    # out_size = (in_size - kernal_size + 2 * padding)/step + 1
    def forward(self, x):
        # x(n, 3, 128, 128)
        x1 = self.cnn_layer1(x)# CNN_Blcok(3, 64, 3, 1, 1, 64)
        x1 = self.relu(x1)
        # x1(n, 64, 128, 128)
        x2 = self.cnn_layer2(x1)# CNN_Blcok(64, 64, 3, 1, 1, 64)
        x2 = x1 + x2
        x2 = self.relu(x2)
        # x2(n, 64, 128, 128)
        x3 = self.cnn_layer3(x2)# CNN_Blcok(64, 128, 3, 2, 1, 128)
        x3 = self.relu(x3)
        # x3(n, 128, 64, 64)
        x4 = self.cnn_layer4(x3)# CNN_Blcok(128, 128, 3, 1, 1, 128)
        x4 = x3 + x4
        x4 = self.relu(x4)
        # x4(n, 128, 64, 64)
        x5 = self.cnn_layer5(x4)# CNN_Blcok(128, 256, 3, 2, 1, 256)
        x5 = self.relu(x5)
        # x5(n, 256, 32, 32)
        x6 = self.cnn_layer6(x5)# CNN_Blcok(256, 256, 3, 1, 1, 256)
        x6 = x5 + x6
        x6 = self.relu(x6)
        # x6(n, 256, 32, 32)
        xout = x6.flatten(1)
        # xout(n, 256 * 32 * 32)
        xout = self.fc_layer(xout)
        # xout(n, 1, 11)
        return xout
    
if __name__ =='__main__':
    doctest.testmod(verbose=True)