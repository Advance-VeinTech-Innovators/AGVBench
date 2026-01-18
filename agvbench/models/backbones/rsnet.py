import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_norm_layer,
                      build_conv_layer, kaiming_init, constant_init)
from mmcv.cnn.bricks import ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def channel_shuffle(x, groups):
    """Channel shuffle operation.

    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W).
        groups (int): Number of groups for channel shuffle.

    Returns:
        torch.Tensor: Shuffled tensor with shape (B, C, H, W).
    """
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ConvBNAct(BaseModule):
    """Convolution + BatchNorm + Activation module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size. Default: 3.
        stride (int): Stride. Default: 1.
        groups (int): Groups for convolution. Default: 1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6'),
                 init_cfg=None):
        super(ConvBNAct, self).__init__(init_cfg)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidual(BaseModule):
    """Inverted Residual block.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride. Default: 1.
        expand_ratio (float): Expansion ratio. Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 expand_ratio=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6'),
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNAct(
                in_channels, hidden_dim, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg))
        
        # Depthwise convolution
        layers.append(ConvBNAct(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            groups=hidden_dim, norm_cfg=norm_cfg, act_cfg=act_cfg))
        
        # Pointwise linear projection
        self.conv = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn_out = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_out(out)
        out = self.bn_out(out)
        if self.use_res_connect:
            return x + out
        else:
            return out


class MAB(BaseModule):
    """Multi-Attention Block (MAB).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6'),
                 init_cfg=None):
        super(MAB, self).__init__(init_cfg)
        self.groups = 2
        
        # Upper Path: 1x1 Conv + BN (y_re)
        self.branch_re = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1])
        
        # Lower Path: expansion -> Dual DW Conv -> y_ss
        self.conv_exp = ConvBNAct(
            in_channels, in_channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Dual DW Conv
        mid_c = (in_channels * 2) // 2
        self.dw_conv1 = nn.Conv2d(
            mid_c, mid_c, 3, padding=1, groups=mid_c, bias=False)
        self.dw_conv2 = nn.Conv2d(
            mid_c, mid_c, 3, padding=1, groups=mid_c, bias=False)
        self.dw_bn = build_norm_layer(norm_cfg, in_channels * 2)[1]
        
        self.conv_ss = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x):
        x_cs = channel_shuffle(x, self.groups)
        
        y_re = self.branch_re(x_cs)
        
        x_exp = self.conv_exp(x_cs)
        c1, c2 = torch.chunk(x_exp, 2, dim=1)
        out_dw1 = self.dw_conv1(c1)
        out_dw2 = self.dw_conv2(c2)
        
        y_ss_mid = torch.cat([out_dw1, out_dw2], dim=1)
        y_ss_mid = self.dw_bn(y_ss_mid)
        y_ss = self.conv_ss(y_ss_mid)
        
        return F.relu(y_re + y_ss)


class RLEB(BaseModule):
    """Region-Level Enhancement Block (RLEB).

    Args:
        in_channels (int): Input channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(RLEB, self).__init__(init_cfg)
        # Side branch F_rpi
        self.side_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg))
        
        # Three independent MABs for different regions
        self.mab_ul = MAB(in_channels, in_channels, norm_cfg=norm_cfg)
        self.mab_ur = MAB(in_channels, in_channels, norm_cfg=norm_cfg)
        self.mab_low = MAB(in_channels, in_channels, norm_cfg=norm_cfg)

    def forward(self, x):
        _, _, h, w = x.size()
        
        # For 7x7 feature map: ul=3x3, ur=3x4, low=4x7
        # Divide into regions according to paper
        f_ul = x[:, :, :3, :3]      # Upper left: 3x3
        f_ur = x[:, :, :3, 3:]       # Upper right: 3x4
        f_low = x[:, :, 3:, :]       # Lower: 4x7
        
        # Parallel MAB processing
        o_ul = self.mab_ul(f_ul)
        o_ur = self.mab_ur(f_ur)
        o_low = self.mab_low(f_low)
        
        # Combine back to original spatial size
        top = torch.cat([o_ul, o_ur], dim=3)
        combined = torch.cat([top, o_low], dim=2)
        
        # Concat with F_rpi
        f_rpi = self.side_branch(x)
        return torch.cat([combined, f_rpi], dim=1)


@BACKBONES.register_module()
class RSNet(BaseBackbone):
    """RSNet: Region-based Shuffle Network.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        stem_channels (int): Stem channels. Default: 16.
        stage_channels (tuple): Channels for each stage. Default: (32, 64, 256).
        expand_ratio (float): Expansion ratio for InvertedResidual. Default: 4.
        out_indices (tuple): Output indices. Default: (3,).
        frozen_stages (int): Frozen stages. Default: -1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: [
                dict(type='Kaiming', layer=['Conv2d']),
                dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ].
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=16,
                 stage_channels=(64, 128, 256),
                 expand_ratio=4,
                 out_indices=(3,),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RSNet, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.stage_channels = stage_channels
        self.expand_ratio = expand_ratio
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval

        assert max(out_indices) < len(stage_channels) + 1  # +1 for RLEB stage

        # Stem (stride=4 to match paper design)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=4, padding=1, bias=False),
            build_norm_layer(norm_cfg, stem_channels)[1],
            build_activation_layer(act_cfg))

        # Stage 1: IR -> DS
        self.stage1_ir = InvertedResidual(
            stem_channels, stage_channels[0], stride=1,
            expand_ratio=expand_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage1_ds = ConvBNAct(
            stage_channels[0], stage_channels[0], kernel_size=3, stride=2,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Stage 2: MAB -> DS
        self.stage2_mab = MAB(
            stage_channels[0], stage_channels[1],
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage2_ds = ConvBNAct(
            stage_channels[1], stage_channels[1], kernel_size=3, stride=2,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Stage 3: MAB -> DS (output: 7x7x256)
        self.stage3_mab = MAB(
            stage_channels[1], stage_channels[2],
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage3_ds = ConvBNAct(
            stage_channels[2], stage_channels[2], kernel_size=3, stride=2,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # RLEB stage
        self.rleb = RLEB(
            stage_channels[2], norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        
        for i in range(self.frozen_stages):
            if i == 0:
                stage = self.stage1_ir
            elif i == 1:
                stage = self.stage2_mab
            elif i == 2:
                stage = self.stage3_mab
            elif i == 3:
                stage = self.rleb
            else:
                break
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _freeze_bn(self):
        """Freeze normalization layers."""
        for m in self.modules():
            if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                m.eval()

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        super(RSNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def forward_features(self, x):
        """Forward features."""
        outs = []
        
        # Stem
        x = self.stem(x)
        
        # Stage 1
        x = self.stage1_ir(x)
        x = self.stage1_ds(x)
        if 0 in self.out_indices:
            outs.append(x)
            if len(self.out_indices) == 1:
                return outs
        
        # Stage 2
        x = self.stage2_mab(x)
        x = self.stage2_ds(x)
        if 1 in self.out_indices:
            outs.append(x)
            if len(self.out_indices) == 1:
                return outs
        
        # Stage 3
        x = self.stage3_mab(x)
        x = self.stage3_ds(x)
        if 2 in self.out_indices:
            outs.append(x)
            if len(self.out_indices) == 1:
                return outs
        
        # RLEB stage
        x = self.rleb(x)
        if 3 in self.out_indices:
            outs.append(x)
        
        return outs

    def forward(self, x):
        """Forward computation."""
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Set training mode."""
        super(RSNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            self._freeze_bn()
