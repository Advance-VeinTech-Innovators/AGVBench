import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import build_activation_layer, build_norm_layer, Conv2d, build_conv_layer
from mmcv.cnn.bricks import DropPath
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def conv_bn(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation=1,
            norm_cfg=dict(type='BN')):
    """Construct a sequential conv and bn.

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        kernel_size (int): kernel_size of the convolution.
        stride (int): stride of the convolution.
        padding (int): stride of the convolution.
        groups (int): groups of the convolution.
        dilation (int): dilation of the convolution. Default to 1.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.

    Returns:
        nn.Sequential(): A conv layer and a batch norm layer.
    """
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False))
    result.add_module('bn', build_norm_layer(norm_cfg, out_channels)[1])
    return result


def conv_bn_relu(in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 dilation=1):
    """Construct a sequential conv, bn and relu.

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        kernel_size (int): kernel_size of the convolution.
        stride (int): stride of the convolution.
        padding (int): stride of the convolution.
        groups (int): groups of the convolution.
        dilation (int): dilation of the convolution. Default to 1.

    Returns:
        nn.Sequential(): A conv layer, batch norm layer and a relu function.
    """

    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    """Fuse the parameters in a branch with a conv and bn.

    Args:
        conv (nn.Conv2d): The convolution module to fuse.
        bn (nn.BatchNorm2d): The batch normalization to fuse.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
        fusing the parameters of conv and bn in one branch.
        The first element is the weight and the second is the bias.
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class LargeKernelConv(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,  # = 1
                 groups,
                 small_kernel,
                 small_kernel_merged=False,
                 init_cfg=None):
        super(LargeKernelConv, self).__init__(init_cfg)
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.small_kernel_merged = small_kernel_merged

        # We assume the conv does not change the feature map size,
        # so padding = k//2.
        # Otherwise, you may configure padding as you wish,
        # and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True)
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.conv_3x3.conv,
                                       self.conv_3x3.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.small_kernel_merged:
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True)

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

        self.small_kernel_merged = True

class ChannelGatingNeck(BaseModule):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].

    Args:
        in_channels (int): Dimension of input features.
        internal_channels (int): Dimension of hidden features.
        out_channels (int): Dimension of output features.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_ratio,
                 drop_path,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ChannelGatingNeck, self).__init__(init_cfg)
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.pw1 = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=1,
            dilation=dilation_ratio[0])
        self.pw2 = conv_bn(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=1,
            dilation=dilation_ratio[-1])
        self.pw3 = conv_bn_relu(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            groups=out_channels)
        self.nonlinear = build_activation_layer(act_cfg)

        self.gating = Gating(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):

        # gating
        g_x = self.gating(x)
        # print("Gating shape:", g_x.shape)

        # channel conv
        c_x = self.preffn_bn(x)
        c_x = self.pw1(c_x)
        # print("Conv 1 shape:", c_x.shape)
        c_x = self.nonlinear(c_x)
        c_x = self.pw2(c_x)
        # print("Conv 2 shape:", c_x.shape)
        
        out = g_x * c_x

        x = self.drop_path(self.pw3(out))
        return x 

class Gating(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='SiLU'),
                 init_cfg=None):
        super(Gating, self).__init__(init_cfg)

        self.gate = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.nonlinear = build_activation_layer(act_cfg)

    def forward(self, x):

        out = self.gate(x)
        out = self.nonlinear(out)
        return out


class StarLKBlock(BaseModule):
    """StarLKBlock for StarLKNet backbone.

    Args:
        in_channels (int): The input channels of the block.
        dw_channels (int): The intermediate channels of the block,
            i.e., input channels of the large kernel convolution.
        block_lk_size (int): size of the super large kernel. Defaults: 31.
        small_kernel (int): size of the parallel small kernel. Defaults: 5.
        drop_path (float): Stochastic depth rate. Defaults: 0.
        small_kernel_merged (bool): Whether to switch the model structure to
            deployment mode (merge the small kernel to the large kernel).
            Default to  False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default to  ``dict(type='ReLU')``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default to  None
    """

    def __init__(self,
                 in_channels,
                 dw_channels,
                 block_lk_size,
                 small_kernel,
                 drop_path,
                 small_kernel_merged=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(StarLKBlock, self).__init__(init_cfg)
        self.dw = dw_channels
        self.inc = in_channels
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = LargeKernelConv(
            in_channels=dw_channels,
            out_channels=dw_channels,
            kernel_size=block_lk_size,
            stride=1,
            groups=dw_channels,
            small_kernel=small_kernel,
            small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = build_activation_layer(act_cfg)
        self.prelkb_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        out = self.prelkb_bn(x)
        out = self.pw1(out)

        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class StarLKNetStage(BaseModule):
    """
    generate StarLKNet blocks for a stage
    return: StarLKNet blocks

    Args:
        channels (int): The input channels of the stage.
        num_blocks (int): The number of blocks of the stage.
        stage_lk_size (int): size of the super large kernel. Defaults: 31.
        drop_path (float): Stochastic depth rate. Defaults: 0.
        small_kernel (int): size of the parallel small kernel. Defaults: 5.
        dw_ratio (float): The intermediate channels
            expansion ratio of the block. Defaults: 1.
        ffn_ratio (float): Mlp expansion ratio. Defaults to 4.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default to  False.
        small_kernel_merged (bool): Whether to switch the model structure to
            deployment mode (merge the small kernel to the large kernel).
            Default to  False.
        norm_intermediate_features (bool): Construct and config norm layer
            or not.
            Using True will normalize the intermediate features for
            downstream dense prediction tasks.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default to  None
    """

    def __init__(
            self,
            channels,
            num_blocks,
            stage_lk_size,
            drop_path,
            small_kernel,
            with_cp=False,  # train with torch.utils.checkpoint to save memory
            small_kernel_merged=False,
            norm_intermediate_features=False,
            norm_cfg=dict(type='BN'),
            init_cfg=None):
        super(StarLKNetStage, self).__init__(init_cfg)
        self.with_cp = with_cp
        self.channels = channels
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path,
                                                         list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size.
            #   You may tune it on your own model.
            starlk_block = StarLKBlock(
                in_channels=channels,
                dw_channels=channels,
                block_lk_size=stage_lk_size,
                small_kernel=small_kernel,
                small_kernel_merged=small_kernel_merged,
                drop_path=block_drop_path)
            blks.append(starlk_block)
        self.blocks = ModuleList(blks)
        if norm_intermediate_features:
            self.norm = build_norm_layer(norm_cfg, channels)[1]
        else:
            self.norm = nn.Identity()

    def forward(self, x):

        for blk in self.blocks:
            if self.with_cp:
                x = checkpoint.checkpoint(blk, x)  # Save training memory
            else:
                x = blk(x)
        return x


@BACKBONES.register_module()
class StarLKNet(BaseBackbone):
    """
    """

    arch_settings = {
        'base':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 512, 1024],
            small_kernel=5,
            dilation_ratio=[1, 3]),
        'large':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[256, 512, 1024, 2048],
            small_kernel=5,
            dilation_ratio=[1, 3]),
        'small':
        dict(
            large_kernel_sizes=[27, 27, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[64, 128, 256, 512],
            small_kernel=3,
            dilation_ratio=[1, 3]),
        'tiny':
        dict(
            large_kernel_sizes=[27, 27, 27, 13],
            layers=[2, 2, 10, 2],
            channels=[32, 64, 128, 256],
            small_kernel=3,
            dilation_ratio=[1, 3]),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 ffn_ratio=1,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 drop_path_rate=0.3,
                 small_kernel_merged=False,
                 norm_intermediate_features=False,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(StarLKNet, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert max(out_indices) < len(arch['layers'])

        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.drop_path_rate = drop_path_rate
        self.small_kernel_merged = small_kernel_merged
        self.norm_eval = norm_eval
        self.norm_intermediate_features = norm_intermediate_features

        base_width = self.arch['channels'][0]
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(self.arch['layers'])
        self.stem = ModuleList([
            conv_bn_relu(
                in_channels=in_channels,
                out_channels=base_width,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1),
            conv_bn_relu(
                in_channels=base_width,
                out_channels=base_width,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1),
            conv_bn_relu(
                in_channels=base_width,
                out_channels=base_width,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=base_width)
        ])
        # stochastic depth. We set block-wise drop-path rate.
        # The higher level blocks are more likely to be dropped.
        # This implementation follows Swin.
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate,
                                             sum(self.arch['layers']))
        ]
        self.stages = ModuleList()
        self.blockneck = ModuleList()
        for stage_idx in range(self.num_stages):
            layer = StarLKNetStage(
                channels=self.arch['channels'][stage_idx],
                num_blocks=self.arch['layers'][stage_idx],
                stage_lk_size=self.arch['large_kernel_sizes'][stage_idx],
                drop_path=dpr[sum(self.arch['layers'][:stage_idx]
                                  ):sum(self.arch['layers'][:stage_idx + 1])],
                small_kernel=self.arch['small_kernel'],
                with_cp=with_cp,
                small_kernel_merged=small_kernel_merged,
                norm_intermediate_features=(stage_idx in out_indices))
            self.stages.append(layer)
            if stage_idx < len(self.arch['layers']) - 1:
                neck = ChannelGatingNeck(
                    in_channels=self.arch['channels'][stage_idx],
                    out_channels=self.arch['channels'][stage_idx + 1],
                    dilation_ratio=self.arch['dilation_ratio'],
                    drop_path=dpr[sum(self.arch['layers'][:stage_idx]
                                  ):sum(self.arch['layers'][:stage_idx + 1])][stage_idx],
                )
                self.blockneck.append(neck)

    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.with_cp:
                x = checkpoint.checkpoint(stem_layer, x)  # save memory
            else:
                x = stem_layer(x)

        #   Need the intermediate feature maps
        outs = []
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            # print("stage_idx:{}".format(stage_idx), x.shape)
            if stage_idx in self.out_indices:
                outs.append(self.stages[stage_idx].norm(x))
            if stage_idx < self.num_stages - 1:
                x = self.blockneck[stage_idx](x)
        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        super(StarLKNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()
        self.small_kernel_merged = True

