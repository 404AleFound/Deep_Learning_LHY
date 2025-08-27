# gan.py -- gan model structre
# Le Jiang 
# 2025/8/16

import torch.nn as nn
import torch
import doctest

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    ---
    TEST
    ---
    >>> input = torch.randn(3, 100)
    >>> model_g = Generator(100)
    >>> output = model_g(input)
    >>> output.shape
    torch.Size([3, 3, 64, 64])
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.transconvd_layer = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=5, 
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),   
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),          
        )

    def forward(self, x):
        y = self.linear_layer(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.transconvd_layer(y)
        return y
    
class Discriminator(nn.Module):
    """
    Input: (batch, 3, 64, 64)
    Output: (batch)
    ---
    TEST
    ---
    >>> input = torch.randn(3, 3, 64, 64)
    >>> model_d = Discriminator(3)
    >>> output = model_d(input)
    >>> output.shape
    torch.Size([3])
    """  
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.Conv2d_layer = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, 
                      stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_dim, feature_dim * 2, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim* 2, feature_dim * 4, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_dim * 8, 1, 
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.Conv2d_layer(x)
        y = y.view(-1)
        return y
    
if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)