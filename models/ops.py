import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, supernet=False):
        super().__init__()

        self.down = ChoiceBlock(in_channels, out_channels, kernel_size=3, stride=2, supernet=supernet)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, supernet=False):
        super().__init__()

        self.up = ChoiceBlock(in_channels, out_channels, kernel_size=2, stride=2, supernet=supernet)

    def forward(self, x):
        return self.up(x)
    

class Cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return torch.cat([x2, x1], dim=1)


class Filter(nn.Module):
    def __init__(self):
        super().__init__()   

    def forward(self, x, channels):
        configurations = channels * np.ones((x.size(0),), int)
        mask = x.new_zeros((x.size(0), x.size(1) + 1))
        mask[np.arange(len(configurations)), configurations] = 1.0
        mask = 1 - mask[:, :x.size(1)].cumsum(1)
        x = x * mask.unsqueeze(2).unsqueeze(3)
        
        return x


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(x)
    
    
class ChoiceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, supernet=False):
        super(ChoiceBlock, self).__init__()
    
        padding = (kernel_size - 1) // 2

        if supernet:
            affine = False
            statistics = True
        else:
            affine = True
            statistics = True
        
        if kernel_size == 2:
            self.choice_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels),
                nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=statistics),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=statistics),
                nn.ReLU(inplace=True)
            )
        else:
            self.choice_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False), 
                nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=statistics),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=statistics),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.choice_block(x)