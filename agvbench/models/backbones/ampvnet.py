import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)

from mmcv.utils.parrots_wrapper import _BatchNorm
from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from mmcv.runner.base_module import BaseModule


class InvertedResidual(BaseModule):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(in_channels * expand_ratio)

        self.proj = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.proj(x)

class AMPVBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(AMPVBlock, self).__init__()
        self.conv = InvertedResidual(in_channels=in_channels, out_channels=out_channels, expand_ratio=4),
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)

        return x


@BACKBONES.register_module()
class AMPVNet(BaseBackbone):
    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 pretrained=None,
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(AMPVNet, self).__init__(init_cfg)
        channels = [64, 128, 256, 512]
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
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
        return AMPVBlock(**kwargs)

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
        super(AMPVNet, self).init_weights(pretrained)
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
        super(AMPVNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()
