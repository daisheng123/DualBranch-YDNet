import torch
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class CBAMBlock(BaseModule):
    """MMSeg官方推荐实现的CBAM模块"""

    def __init__(self,
                 in_channels,
                 reduction_ratio=16,
                 kernel_size=7,
                 conv_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        # 通道注意力
        self.channel_attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            build_conv_layer(
                conv_cfg,
                in_channels,
                in_channels // reduction_ratio,
                kernel_size=1),
            torch.nn.ReLU(inplace=True),
            build_conv_layer(
                conv_cfg,
                in_channels // reduction_ratio,
                in_channels,
                kernel_size=1),
            torch.nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = torch.nn.Sequential(
            build_conv_layer(
                conv_cfg,
                2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # 通道分支
        ca = self.channel_attention(x)
        x_ca = ca * x

        # 空间分支
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return sa * x_ca