# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import warnings

import torch
import torch.nn as nn

from models.common import Conv


class Bottleneck(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True,
                 g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in + cv2out, cv3out, 1)
        self.m = nn.Sequential(*[Bottleneck(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPF, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
