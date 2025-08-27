# conv.py -- encoder-decoder based on conv2d
# Le Jiang
# 2025/8/24

import torch
from torch import nn
import doctest

class Autoencoder_Conv2d(nn.Module):
    '''
    ---
    TEST
    ---
    >>> x = torch.randn(3, 64, 64)
    >>> my_model = Autoencoder_Conv2d()
    >>> output = my_model(x)
    >>> output.shape == x.shape
    True
    '''
    def __init__(self):
        super(Autoencoder_Conv2d, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':
    doctest.testmod(verbose=True)