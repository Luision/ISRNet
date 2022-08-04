import torch
import torch.nn as nn
from util import swap_axis
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def split(x):
    n, c, w, h = x.shape
    x1 = x[:, :, 0:w:2, 0:h:2]
    x2 = x[:, :, 0:w:2, 1:h:2]
    x3 = x[:, :, 1:w:2, 0:h:2]
    x4 = x[:, :, 1:w:2, 1:h:2]
    return torch.cat([x1, x2], dim=1), torch.cat([x3, x4], dim=1)

def concat(y1, y2, shape):
    n, c, w, h = shape
    x = torch.ones((n, c, w, h)).cuda()
    x[:, :, 0:w:2, 0:h:2] = y1[:, :1, :, :]
    x[:, :, 0:w:2, 1:h:2] = y1[:, 1:, :, :]
    x[:, :, 1:w:2, 0:h:2] = y2[:, :1, :, :]
    x[:, :, 1:w:2, 1:h:2] = y2[:, 1:, :, :]
    return x

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.f = nn.Sequential(*[
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=3, padding=1, stride=1, bias=True),
            nn.Conv2d(in_channels=96, out_channels=16, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
            nn.Conv2d(in_channels=16, out_channels=96, kernel_size=1, stride=1, bias=True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, stride=1, bias=True),
        ])

    def forward(self, x):
        return self.f(x)

class InvRescaleNet(nn.Module):
    def __init__(self):
        super(InvRescaleNet, self).__init__()
        self.downsample = nn.AvgPool2d(2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

        self.F = block()
        self.G = block()
        self.H = block()

    def forward(self, x, rev=False):
        if not rev:
            x = swap_axis(x)
            x = self.upsample(x)
            shape = x.shape
            x1, x2 = split(x)
            x1 = x1 + self.F(x2)
            x2 = x2 + self.G(x1)
            x1 = x1 + self.H(x2)
            y = concat(x1, x2, shape)
            y = swap_axis(y)
        else:

            x = swap_axis(x)
            shape = x.shape
            x1, x2 = split(x)
            x1 = x1 - self.H(x2)
            x2 = x2 - self.G(x1)
            x1 = x1 - self.F(x2)
            y = concat(x1, x2, shape)
            y = self.downsample(y)
            y = swap_axis(y)

        return y

def weights_init_I(m):
    """ initialize weights of the upsampler  """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)

