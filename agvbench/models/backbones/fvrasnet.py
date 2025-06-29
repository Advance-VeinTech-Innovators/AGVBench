import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)

from mmcv.utils.parrots_wrapper import _BatchNorm
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


# FVRAS-Net: An Embedded Finger-Vein Recognition and AntiSpoofing System Using a Unified CNN
@BACKBONES.register_module()
class FVRASNet(BaseBackbone):
    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 pretrained=None,
                 out_indices=(0, 1, 2,),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(FVRASNet, self).__init__(init_cfg)
        channels = [64, 128, 256]
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(),
        )
        self.conv_block = []
        _in_channels = stem_channels
        for i in range(0, len(channels)):
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
        if self.frozen_stages >= 0:
            if self.deep_stem:
                # self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                # self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            # m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _freeze_bn(self):
        """ keep normalization layer freezed. """
        for m in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                m.eval()

    def _unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                m.train()

    def init_weights(self, pretrained=None):
        super(FVRASNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, block_name in enumerate(self.conv_block):
            ras_block = getattr(self, block_name)
            x = ras_block(x)
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
        return outs

    def train(self, mode=True):
        super(FVRASNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        return x