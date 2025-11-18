import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class SegNext_Attention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv0_1 = nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3), groups=in_channels)
        self.conv0_2 = nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), groups=in_channels)

        self.conv1_1 = nn.Conv2d(in_channels, in_channels, (1, 11), padding=(0, 5), groups=in_channels)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, (11, 1), padding=(5, 0), groups=in_channels)

        self.conv2_1 = nn.Conv2d(in_channels, in_channels, (1, 21), padding=(0, 10), groups=in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, (21, 1), padding=(10, 0), groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
