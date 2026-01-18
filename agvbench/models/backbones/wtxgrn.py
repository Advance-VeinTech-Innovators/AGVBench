import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer,
                      build_conv_layer, kaiming_init, constant_init)
from mmcv.cnn.bricks import DropPath
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from .base_backbone import BaseBackbone
from ..builder import BACKBONES
from agvbench.third_party.wtxgrn_util import WTConv2d
from agvbench.third_party.xgru_block import MultiDirectionGRUBlock


def to_ntuple(n):
    """Convert to n-tuple."""
    def parse(x):
        if isinstance(x, (int, float)):
            return tuple([x] * n)
        elif isinstance(x, (list, tuple)):
            assert len(x) == n, f'Length mismatch: {len(x)} vs {n}'
            return tuple(x)
        else:
            raise TypeError(f'Cannot convert {type(x)} to {n}-tuple')
    return parse


class Downsample(BaseModule):
    """Downsample module for feature map size reduction."""

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, init_cfg=None):
        super(Downsample, self).__init__(init_cfg)
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            self.pool = nn.AvgPool2d(
                kernel_size=2, stride=avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class WTConvBlock(BaseModule):
    """WTConvNeXt Block with Wavelet Transform Convolution.

    Args:
        in_channels (int): Block input channels.
        out_channels (int): Block output channels (same as in_channels if None).
        kernel_size (int): Depthwise convolution kernel size. Default: 5.
        stride (int): Stride of depthwise convolution. Default: 1.
        dilation (int or tuple): Dilation of convolution. Default: (1, 1).
        expansion (float): Channel expansion ratio. Default: 4.
        ls_init_value (float): Layer-scale init values. Default: 1e-6.
        drop_path (float): Stochastic depth probability. Default: 0.
        wt_levels (int): Number of WT levels for WTConv. Default: 1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=5,
                 stride=1,
                 dilation=(1, 1),
                 expansion=4,
                 ls_init_value=1e-6,
                 drop_path=0.,
                 wt_levels=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(WTConvBlock, self).__init__(init_cfg)
        out_channels = out_channels or in_channels
        dilation = to_ntuple(2)(dilation)
        
        self.norm_layer = build_norm_layer(norm_cfg, in_channels)[1]
        self.act_layer = build_activation_layer(act_cfg)
        self.wt_dwconv = WTConv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, wt_levels=wt_levels)
        self.hidden_conv = nn.Conv2d(
            in_channels, int(in_channels * expansion), kernel_size=1)
        self.conv_out = nn.Conv2d(
            int(in_channels * expansion), out_channels, kernel_size=1)

        self.gamma = nn.Parameter(
            ls_init_value * torch.ones(out_channels)) if ls_init_value is not None else None
        
        if in_channels != out_channels or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_channels, out_channels, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_x1=False):
        shortcut = x
        x1 = self.wt_dwconv(x)
        x2 = self.norm_layer(x1)
        x2 = self.hidden_conv(x2)
        x2 = self.act_layer(x2)
        x2 = self.conv_out(x2)
        
        if self.gamma is not None:
            x2 = x2.mul(self.gamma.reshape(1, -1, 1, 1))
        
        x2 = self.drop_path(x2) + self.shortcut(shortcut)
        return (x1, x2) if return_x1 else x2


class ChannelMixer(BaseModule):
    """Channel mixer: feature maps -> patch embeddings.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        dw_stride (int): Downsample stride.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dw_stride,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ChannelMixer, self).__init__(init_cfg)
        self.dw_stride = dw_stride
        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(
            kernel_size=dw_stride, stride=dw_stride)
        # Use nn.LayerNorm directly for sequence data (B, L, D)
        eps = norm_cfg.get('eps', 1e-6) if isinstance(norm_cfg, dict) else 1e-6
        self.ln = nn.LayerNorm(out_channels, eps=eps)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, x_g):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        return x + x_g


class SpatialMixer(BaseModule):
    """Spatial mixer: patch embeddings -> feature maps.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        up_stride (int): Upsample stride.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 up_stride,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(SpatialMixer, self).__init__(init_cfg)
        self.conv_project = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=up_stride, stride=up_stride)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, x_w, seqlens):
        B, L, D = x.shape
        H, W = seqlens
        assert L == H * W
        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.act(self.bn(self.conv_project(x)))
        return x + x_w


class WTConvGRUBlock(BaseModule):
    """WTConvGRU Block combining WTConv and GRU.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels (same as in_channels if None).
        embed_dim (int): Embedding dimension. Default: 192.
        kernel_size (int): Kernel size. Default: 5.
        stride (int): Stride. Default: 1.
        dw_stride (int): Downsample stride. Default: 4.
        wt_levels (int): WT levels. Default: 1.
        drop_path (float): Drop path rate. Default: 0.
        seqlens (tuple): Sequence lengths (H, W). Default: (8, 8).
        conv_kind (str): Convolution kind. Default: '2d'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 embed_dim=192,
                 kernel_size=5,
                 stride=1,
                 dw_stride=4,
                 wt_levels=1,
                 drop_path=0.,
                 seqlens=(8, 8),
                 conv_kind='2d',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(WTConvGRUBlock, self).__init__(init_cfg)
        self.seqlens = seqlens
        out_channels = out_channels or in_channels
        
        self.wtconv_block = WTConvBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, wt_levels=wt_levels,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.gru_block = MultiDirectionGRUBlock(
            dim=embed_dim, directions='default', expansion=2, num_heads=4,
            drop_path=drop_path, conv_kind=conv_kind, seqlens=seqlens)
        
        self.channelmixer = ChannelMixer(
            in_channels=in_channels, out_channels=embed_dim,
            dw_stride=dw_stride // stride, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.spatialmixer = SpatialMixer(
            in_channels=embed_dim, out_channels=out_channels,
            up_stride=dw_stride // stride, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.fusion_block = WTConvBlock(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, wt_levels=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x_g, x_w):
        # x_g: [B,L,D], x_w: [B,C,H,W]
        x_w1, x_w2 = self.wtconv_block(x_w, return_x1=True)
        x_wg = self.channelmixer(x_w1, x_g)  # [B,L,D]
        x_g = self.gru_block(x_wg)  # [B,L,D]
        x_gw = self.spatialmixer(x_g, x_w2, seqlens=self.seqlens)  # [B,C,H,W]
        x_w = self.fusion_block(x_gw, return_x1=False)  # [B,C,H,W]
        return x_g, x_w


class WTConvGRUStage(BaseModule):
    """WTConvGRU Stage containing multiple blocks.

    Args:
        in_channels (int): Input channels.
        embed_dim (int): Embedding dimension. Default: 192.
        depth (int): Number of blocks. Default: 2.
        kernel_size (int): Kernel size. Default: 5.
        dw_stride (int): Downsample stride. Default: 4.
        wt_levels (int): WT levels. Default: 1.
        drop_path (float): Drop path rate. Default: 0.
        seqlens (tuple): Sequence lengths. Default: (8, 8).
        conv_kind (str): Convolution kind. Default: '2d'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 embed_dim=192,
                 depth=2,
                 kernel_size=5,
                 dw_stride=4,
                 wt_levels=1,
                 drop_path=0.,
                 seqlens=(8, 8),
                 conv_kind='2d',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(WTConvGRUStage, self).__init__(init_cfg)
        blocks = []
        for i in range(depth):
            stride = 2 if i == depth - 1 and dw_stride > 1 else 1
            out_channels = in_channels * 2 if i == depth - 1 else in_channels
            block = WTConvGRUBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                wt_levels=wt_levels,
                drop_path=drop_path,
                dw_stride=dw_stride,
                seqlens=seqlens,
                conv_kind=conv_kind,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            blocks.append(block)
            in_channels = out_channels
        self.blocks = ModuleList(blocks)

    def forward(self, x_g, x_w):
        for block in self.blocks:
            x_g, x_w = block(x_g, x_w)
        return x_g, x_w


class StemBlock(BaseModule):
    """Stem block for initial feature extraction.

    Args:
        in_channels (int): Input channels.
        base_channels (int): Base channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 base_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(StemBlock, self).__init__(init_cfg)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7,
                     stride=2, padding=3, bias=False),
            build_norm_layer(norm_cfg, base_channels)[1],
            build_activation_layer(act_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.dw_stride = 4

    def forward(self, x):
        return self.proj(x)


@BACKBONES.register_module()
class WTxGRN(BaseBackbone):
    """WTxGRN: Wavelet Transform Convolution with GRU Network.

    Args:
        arch (str or dict): Architecture config. Can be 'tiny', 'small', 'base', 'large'.
        in_channels (int): Number of input channels. Default: 3.
        img_size (int): Input image size. Default: 224.
        patch_size (int): Patch size. Default: 16.
        embed_dim (int): Embedding dimension. Default: 192.
        base_channels (int): Base channels. Default: 64.
        depths (tuple): Number of blocks in each stage. Default: (2, 2, 2).
        channel_ratio (float): Channel ratio. Default: 2.
        expand (int): Expansion ratio. Default: 2.
        num_heads (int): Number of attention heads. Default: 4.
        kernel_size (int): Kernel size. Default: 5.
        wt_levels (int): WT levels. Default: 1.
        drop_path_rate (float): Drop path rate. Default: 0.0.
        conv_kind (str): Convolution kind. Default: 'wt2d'.
        out_indices (tuple): Output indices. Default: (2,).
        frozen_stages (int): Frozen stages. Default: -1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_eval (bool): Whether to set norm layers to eval mode.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: [
                dict(type='Kaiming', layer=['Conv2d']),
                dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ].
    """

    arch_settings = {
        'tiny': dict(
            embed_dim=192,
            base_channels=64,
            depths=(2, 2, 2),
            channel_ratio=1,
            wt_levels=1),
        'small': dict(
            embed_dim=192,
            base_channels=64,
            depths=(2, 2, 2),
            channel_ratio=1,
            wt_levels=1),
        'base': dict(
            embed_dim=256,
            base_channels=64,
            depths=(2, 2, 2),
            channel_ratio=1.5,
            wt_levels=1),
        'large': dict(
            embed_dim=384,
            base_channels=64,
            depths=(3, 3, 3),
            channel_ratio=2,
            wt_levels=1),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 img_size=224,
                 patch_size=16,
                 embed_dim=None,
                 base_channels=None,
                 depths=None,
                 channel_ratio=None,
                 expand=2,
                 num_heads=4,
                 kernel_size=5,
                 wt_levels=1,
                 drop_path_rate=0.0,
                 conv_kind='wt2d',
                 out_indices=(2,),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(WTxGRN, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        # Use arch settings if not specified
        embed_dim = embed_dim or arch.get('embed_dim', 192)
        base_channels = base_channels or arch.get('base_channels', 64)
        depths = depths or arch.get('depths', (2, 2, 2))
        channel_ratio = channel_ratio or arch.get('channel_ratio', 2)
        wt_levels = wt_levels or arch.get('wt_levels', 1)

        self.arch = arch
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.depths = depths
        self.channel_ratio = channel_ratio
        self.expand = expand
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.wt_levels = wt_levels
        self.drop_path_rate = drop_path_rate
        self.conv_kind = conv_kind
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval

        self.num_stages = len(depths)
        assert self.num_stages <= 4
        assert max(out_indices) < self.num_stages

        # Stem stage
        self.stem_block = StemBlock(
            in_channels=in_channels,
            base_channels=base_channels,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        dw_stride = self.stem_block.dw_stride

        # GRU patch embedding
        self.gru_patch_embed = nn.Conv2d(
            base_channels, embed_dim,
            kernel_size=patch_size // dw_stride,
            stride=patch_size // dw_stride)
        self.seqlens = [img_size // patch_size] * 2

        # Initial conv projection
        conv_chans = int(base_channels * channel_ratio)
        self.wtconv_0 = nn.Conv2d(base_channels, conv_chans, kernel_size=1, stride=1)

        # Stochastic depth
        depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stages
        self.stages = ModuleList()
        stage_idx = 0
        for i in range(self.num_stages):
            stage_in_channels = int(conv_chans * 2 ** i)
            stage_dw_stride = int(dw_stride // 2 ** i)
            stage_dpr = dpr[sum(depths[:i]):sum(depths[:i+1])]
            
            stage = WTConvGRUStage(
                in_channels=stage_in_channels,
                embed_dim=embed_dim,
                depth=depths[i],
                kernel_size=kernel_size,
                dw_stride=stage_dw_stride,
                wt_levels=wt_levels,
                drop_path=stage_dpr[0] if isinstance(stage_dpr, list) else stage_dpr,
                seqlens=self.seqlens,
                conv_kind=conv_kind,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.stages.append(stage)
            stage_idx += 1

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.stem_block.eval()
            for param in self.stem_block.parameters():
                param.requires_grad = False
        
        for i in range(self.frozen_stages):
            stage = self.stages[i]
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
        super(WTxGRN, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def forward_features(self, x):
        """Forward features."""
        # Stem stage
        x_base = self.stem_block(x)  # [N, C, H, W]

        # GRU and Conv branches
        x_g = self.gru_patch_embed(x_base).flatten(2).transpose(1, 2)  # [N, L, D]
        x_w = self.wtconv_0(x_base)  # [N, C, H, W]

        outs = []
        for i, stage in enumerate(self.stages):
            x_g, x_w = stage(x_g, x_w)
            if i in self.out_indices:
                outs.append(x_w)
                if len(self.out_indices) == 1:
                    return outs
        return outs

    def forward(self, x):
        """Forward computation."""
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Set training mode."""
        super(WTxGRN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            self._freeze_bn()