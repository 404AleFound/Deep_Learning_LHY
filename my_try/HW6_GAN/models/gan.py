import torch.nn as nn

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()
    
        #input: 输入随机一维向量 (batch, 100) 随机生成噪点数据 -> (batch, 64 * 8 * 4 * 4)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        # y.view(y.size(0), -1, 4, 4) -> 转成 (batch, feature_dim * 8, 4, 4)
        # 上采样并提取特征：逐步将channel中的特征信息转到 height and width 维度
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),               # out_put -> (batch, feature_dim * 4, 8, 8)     
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),               # out_put -> (batch, feature_dim * 2, 16, 16)     
            self.dconv_bn_relu(feature_dim * 2, feature_dim),                   # out_put -> (batch, feature_dim, 32, 32)     
        )
        # out_put -> (batch, 3, 64, 64) channel dim=1
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()   
            
        )

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),        # 双倍 height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

class Discriminator(nn.Module):
    """
    输入: (batch, 3, 64, 64)
    输出: (batch)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()
            
        # input: (batch, 3, 64, 64)
        """
        设置Discriminator的注意事项:
            在WGAN中需要移除最后一层 sigmoid
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), # output -> (batch, 64, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   # output -> (batch, 128, 32, 32)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               # output -> (batch, 256, 32, 32)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               # output -> (batch, 512, 32, 32)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),  # output -> (batch, 1, 1, 1)
            nn.Sigmoid() 
        )
        
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        设置Discriminator的注意事项:
            在WGAN-GP中不能使用 nn.Batchnorm， 需要使用 nn.InstanceNorm2d 替代
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y
