# fcn.py -- 
# Le Jiang
# 2025/8/25

import torch
import torch.nn as nn
import doctest

class Autoencoder_Fcn(nn.Module):
    '''
    ---
    TEST
    ---
    >>> x = torch.randn(64*64*3)
    >>> autoencoder_fcn = Autoencoder_Fcn()
    >>> output = autoencoder_fcn(x)
    >>> x.shape == output.shape
    True
    
    '''
    def __init__(self):
        super(Autoencoder_Fcn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 64 * 2),
            nn.ReLU(),
            nn.Linear(64 * 2, 64 * 1),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64 * 1, 64 * 2),
            nn.ReLU(),
            nn.Linear(64 * 2, 64 * 64 * 3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    doctest.testmod(verbose=True)