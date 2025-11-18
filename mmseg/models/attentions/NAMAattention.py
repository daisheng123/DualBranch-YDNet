import torch.nn as nn
import torch
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


@MODELS.register_module()
class NAMAttentionblock(nn.Module):
    def __init__(self, in_channels):
        super(NAMAttentionblock, self).__init__()
        self.Channel_Att = nn.Sequential(*(Channel_Att(in_channels)for _ in range(1)))

    def forward(self, x):
        # print(x.device)
        #
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x_out1 = self.Channel_Att(x)

        return x_out1