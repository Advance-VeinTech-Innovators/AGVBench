import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from thop import profile
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, LayerNorm2d, to_ntuple
from .wtxgrn_util import WTConv2d
from .xgru_block import MultiDirectionGRUBlock


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chans != out_chans:
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class WTConvBlock(nn.Module):
    """ WTConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: Optional[int] = None,
            kernel_size: int = 5,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            expansion: float = 4,
            ls_init_value: Optional[float] = 1e-6,
            drop_path: float = 0.,
            wt_levels: int = 1,
    ):
        """

        Args:
            in_chans: Block input channels.
            out_chans: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            expansion: channel expansion ratio.
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            drop_path: Stochastic depth probability.
            wt_levels: Number of WT levels for WTConv.
        """
        super().__init__()
        out_chans = out_chans or in_chans
        dilation = to_ntuple(2)(dilation)
        self.act_layer = nn.GELU()
        self.norm_layer = LayerNorm2d(in_chans)
        self.wt_dwconv = WTConv2d(in_chans, in_chans, kernel_size=kernel_size, stride=stride, wt_levels=wt_levels)
        self.hidden_conv = nn.Conv2d(in_chans, int(in_chans * expansion), kernel_size=1)
        self.conv_out = nn.Conv2d(int(in_chans * expansion), out_chans, kernel_size=1)

        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chans)) if ls_init_value is not None else None
        if in_chans != out_chans or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chans, out_chans, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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


class ChannelMixer(nn.Module):
    """
    feature maps -> patch embeddings
    """

    def __init__(self, in_chans, out_chans, dw_stride):
        super().__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = nn.LayerNorm(out_chans, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x, x_g):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        return x + x_g


class SpatialMixer(nn.Module):
    """
    patch embeddings -> feature maps
    """

    def __init__(self, in_chans, out_chans, up_stride):
        super().__init__()

        # self.upsampler = DySample(in_chans=in_chans, out_chans=out_chans, scale=up_stride, dyscope=True)
        # self.conv_project = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.conv_project = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=up_stride, stride=up_stride)
        self.bn = nn.BatchNorm2d(out_chans, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x, x_w, seqlens):
        B, L, D = x.shape
        H, W = seqlens
        assert L == H * W
        # [N, 64, D] -> [N, 64, D] -> [N, D, 64] -> [N, D, 8, 8]
        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.act(self.bn(self.conv_project(x)))
        # return F.interpolate(x, size=(H * self.up_stride, W * self.up_stride))
        return x + x_w


class WTConvGRUBlock(nn.Module):
    def __init__(
            self,
            in_chans,
            out_chans=None,
            embed_dim=192,
            kernel_size=5,
            stride=1,
            dw_stride=4,
            wt_levels=1,
            drop_path=0.,
            seqlens=(8, 8),
            conv_kind="2d",
    ):
        super().__init__()
        self.seqlens = seqlens
        self.wtconv_block = WTConvBlock(in_chans=in_chans, out_chans=out_chans, kernel_size=kernel_size,
                                        stride=stride, wt_levels=wt_levels)
        self.gru_block = MultiDirectionGRUBlock(dim=embed_dim, directions="default", expansion=2, num_heads=4,
                                                drop_path=drop_path, conv_kind=conv_kind, seqlens=seqlens)

        self.channelmixer = ChannelMixer(in_chans=in_chans, out_chans=embed_dim, dw_stride=dw_stride // stride)
        self.spatialmixer = SpatialMixer(in_chans=embed_dim, out_chans=out_chans, up_stride=dw_stride // stride)
        self.fusion_block = WTConvBlock(in_chans=out_chans, out_chans=out_chans, kernel_size=3, stride=1, wt_levels=1)

    def forward(self, x_g, x_w):
        # x_g: [B,L,D], x_w: [B,C,H,W]
        x_w1, x_w2 = self.wtconv_block(x_w, return_x1=True)  # [B,C,H,W], [B,C,H,W]
        x_wg = self.channelmixer(x_w1, x_g)  # [B,L,D]
        x_g = self.gru_block(x_wg)  # [B,L,D]
        x_gw = self.spatialmixer(x_g, x_w2, seqlens=self.seqlens)  # [B,C,H,W]
        x_w = self.fusion_block(x_gw, return_x1=False)  # [B,C,H,W]
        return x_g, x_w


class WTConvGRUStage(nn.Module):
    def __init__(
            self,
            in_chans,
            embed_dim=192,
            depth=2,
            kernel_size=5,
            dw_stride=4,
            wt_levels=1,
            drop_path=0.,
            seqlens=(8, 8),
            conv_kind="2d",
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            stride = 2 if i == depth - 1 and dw_stride > 1 else 1
            out_chans = in_chans * 2 if i == depth - 1 else in_chans
            block = WTConvGRUBlock(
                in_chans=in_chans,
                out_chans=out_chans,
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                wt_levels=wt_levels,
                drop_path=drop_path,
                dw_stride=dw_stride,
                seqlens=seqlens,
                conv_kind=conv_kind
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_g, x_w):
        for i, block in enumerate(self.blocks):
            x_g, x_w = block(x_g, x_w)
        return x_g, x_w


class StemBlock(nn.Module):
    def __init__(self, in_chans, base_chans):
        super(StemBlock, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, base_chans, kernel_size=7, stride=2, padding=3, bias=False),  # 1/2
            nn.BatchNorm2d(base_chans),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [32, 32]
        )

        self.dw_stride = 4

    def forward(self, x):
        x = self.proj(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)


class WTxGRN(nn.Module):
    def __init__(
            self,
            img_size=128,
            in_chans=1,
            patch_size=16,
            embed_dim=192,
            base_chans=64,
            depths=(2, 2, 2),
            num_classes=600,
            channel_ratio=2,
            expand=2,
            num_heads=4,
            kernel_size=5,
            wt_levels=1,
            drop_path_rate=0.0,
            gru_pool="bilateral_flatten",
            conv_kind="wt2d"
    ):

        super().__init__()
        self.num_stages = len(depths)
        assert self.num_stages <= 4
        assert gru_pool in ["bilateral_flatten", "bilateral_avg"]
        depth = sum(depths)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.base_chans = base_chans
        self.num_heads = num_heads
        self.expand = expand
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.in_chans = in_chans
        self.channel_ratio = channel_ratio
        self.gru_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Stem stage: get the feature maps by conv block (copied form ResNet.py)
        self.stem_block = StemBlock(in_chans=in_chans, base_chans=base_chans)
        dw_stride = self.stem_block.dw_stride

        self.gru_patch_embed = nn.Conv2d(base_chans, embed_dim, kernel_size=patch_size // dw_stride,
                                         stride=patch_size // dw_stride)
        self.seqlens = [img_size // patch_size] * 2

        conv_chans = int(base_chans * self.channel_ratio)
        self.wtconv_0 = nn.Conv2d(base_chans, conv_chans, kernel_size=1, stride=1)

        stages = []
        for i in range(self.num_stages):
            stage_in_chans = int(conv_chans * 2 ** i)
            stage_dw_stride = int(dw_stride // 2 ** i)
            stage = WTConvGRUStage(
                in_chans=stage_in_chans,
                embed_dim=embed_dim,
                depth=depths[i],
                kernel_size=kernel_size,
                dw_stride=stage_dw_stride,
                wt_levels=wt_levels,
                drop_path=drop_path_rate,
                seqlens=self.seqlens,
                conv_kind=conv_kind
            )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

        # Classifier head
        self.gru_pool = gru_pool
        if self.gru_pool == "bilateral_flatten":
            gru_head_dim = embed_dim * 2
        else:
            gru_head_dim = embed_dim
        self.gru_final_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.gru_cls_head = nn.Linear(gru_head_dim, num_classes)

        self.wtconv_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_cls_head = nn.Linear(int(base_chans * channel_ratio * 2 ** self.num_stages), num_classes)

        self.apply(_init_weights)

    def forward(self, x, return_total=True):
        # stem stage [N, 1, 128, 128] -> [N, bc, 32, 32]
        x_base = self.stem_block(x)

        x_g = self.gru_patch_embed(x_base).flatten(2).transpose(1, 2)
        x_w = self.wtconv_0(x_base)

        for i, stage in enumerate(self.stages):
            x_g, x_w = stage(x_g, x_w)

        # gru classification
        x_g = self.gru_final_norm(x_g)
        if self.gru_pool == "bilateral_avg":
            x_g = (x_g[:, 0] + x_g[:, -1]) / 2
        elif self.gru_pool == "bilateral_flatten":
            x_g = torch.cat([x_g[:, 0], x_g[:, -1]], dim=1)
        else:
            raise NotImplementedError(f"pooling '{self.gru_pool}' is not implemented")
        gru_cls = self.gru_cls_head(x_g)

        # conv classification
        x_w = self.wtconv_pool(x_w).flatten(1)
        wtconv_cls = self.conv_cls_head(x_w)

        if return_total:
            return gru_cls + wtconv_cls

        return gru_cls, wtconv_cls


def WTGRUNet_L(num_classes, pretrained_ckpt=None, device="cuda", logger=None, **kwargs):
    model = WTxGRN(img_size=128, in_chans=1, patch_size=16, embed_dim=384, base_chans=64, depths=(3, 3, 3),
                     num_classes=num_classes, channel_ratio=2, expand=2, num_heads=4, gru_pool="bilateral_flatten",
                     **kwargs).to(device)
    macs, params = profile(model, inputs=(torch.randn(1, 1, 128, 128).to(device),),
                           verbose=False)
    flops = macs / 1e9
    params = params / 1e6
    logger.info(f"params:{params}M  flops:{flops}G")
    return load_pretrained_ckpt(model, device=device, pretrained_ckpt=pretrained_ckpt, logger=logger)


def WTGRUNet_M(num_classes, pretrained_ckpt=None, device="cuda", logger=None, **kwargs):
    model = WTxGRN(img_size=128, in_chans=1, patch_size=16, embed_dim=256, base_chans=64, depths=(2, 2, 2),
                     num_classes=num_classes, channel_ratio=1.5, expand=2, num_heads=4, gru_pool="bilateral_flatten",
                     **kwargs).to(device)
    macs, params = profile(model, inputs=(torch.randn(1, 1, 128, 128).to(device),),
                           verbose=False)
    flops = macs / 1e9
    params = params / 1e6
    logger.info(f"params:{params}M  flops:{flops}G")
    return load_pretrained_ckpt(model, device=device, pretrained_ckpt=pretrained_ckpt, logger=logger)


def WTGRUNet_S(num_classes, pretrained_ckpt=None, device="cuda", logger=None, **kwargs):
    model = WTxGRN(img_size=128, in_chans=1, patch_size=16, embed_dim=192, base_chans=64, depths=(2, 2, 2),
                     num_classes=num_classes, channel_ratio=1, expand=2, num_heads=4, gru_pool="bilateral_flatten",
                     **kwargs).to(device)
    macs, params = profile(model, inputs=(torch.randn(1, 1, 128, 128).to(device),),
                           verbose=False)
    flops = macs / 1e9
    params = params / 1e6
    logger.info(f"params:{params}M  flops:{flops}G")
    return load_pretrained_ckpt(model, device=device, pretrained_ckpt=pretrained_ckpt, logger=logger)


def load_pretrained_ckpt(model, device="cuda", pretrained_ckpt=None, logger=None):
    if pretrained_ckpt is None:
        return model
    else:
        if logger is None:
            print("warning: logger is None, the information may not be visible")
            logger = logging.getLogger(__name__)
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        if len(missing_keys) != 0:
            logger.warning(f'Warning:Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.warning(f'Warning:Unexpected keys in source state dict: {unexpected_keys}')
    return model
