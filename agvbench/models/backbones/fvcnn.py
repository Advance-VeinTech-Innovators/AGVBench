import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,

                 ):
        super(ConvBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.maxpool(x)

        return x

# Convolutional Neural Network for Finger-Vein-Based Biometric Identification
@BACKBONES.register_module()
class FVCNN(BaseBackbone):
    def __init__(self,
                 pretrained=None,
                 in_channels=3,
                 blocks=3,
                 style='pytorch',
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super(FVCNN, self).__init__(init_cfg)
        self.block = blocks
        self.style = style
        self.frozen_stages = frozen_stages
        channels = [128, 512, 768]

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=4),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=500, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.conv_block = []
        _in_channels = in_channels
        for i in range(0, self.block):
            ras_layer = self.make_conv_block(
                in_channels=_in_channels,
                out_channels=channels[i],
            )
            _in_channels = channels[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, ras_layer)
            self.conv_block.append(layer_name)

        self.init_weights(pretrained=pretrained)

    def make_conv_block(self, **kwargs):
        return ConvBlock(**kwargs)

    def _freeze_stages(self):
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(FVCNN, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def forward(self, x):

        for i, block_name in enumerate(self.conv_block):
            ras_block = getattr(self, block_name)
            x = ras_block(x)

        x = [self.tail(x)]
        return x