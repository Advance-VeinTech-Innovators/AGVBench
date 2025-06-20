import math
from typing import Sequence, Tuple, Union
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, \
                                       uniform_init, xavier_init
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from agvbench.utils import get_root_logger, print_log
from ..utils import to_2tuple
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

# from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    
class RetNetRelPos2d(BaseModule):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 initial_value,
                 heads_range
                ):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super(RetNetRelPos2d, self).__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None]  #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
    
class VisionRetentionChunk(BaseModule):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 value_factor=1
                ):
        super(VisionRetentionChunk, self).__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
class VisionRetentionAll(BaseModule):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 value_factor=1
                ):
        super(VisionRetentionAll, self).__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)

        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class FeedForwardNetwork(BaseModule):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_cfg=dict(type='GELU'),
        dropout=0.0,
        act_dropout=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        subln=False,
        subconv=True,
        init_cfg=None,
        ):
        super(FeedForwardNetwork, self).__init__(init_cfg)

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.act_fn = build_activation_layer(act_cfg)
        self.act_dropout_module = torch.nn.Dropout(act_dropout)

        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim)

        # self.ffn_layernorm = build_norm_layer(norm_cfg, self.ffn_dim) if subln else None
        if subln:
            self.ffn_norm_name, ffn_norm = build_norm_layer(
                        norm_cfg, ffn_dim, postfix=2)
            self.add_module(self.ffn_norm_name, ffn_norm)
        else:
            self.ffn_norm = nn.Identity()

        self.dwconv = DWConv2d(self.ffn_dim, 3, 1, 1) if subconv else None

    @property
    def ffn_norm(self):
        return getattr(self, self.ffn_norm_name)

    def reset_parameters(self):

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

        if self.ffn_norm is not None:
            self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.act_dropout_module(x)

        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)

        if self.ffn_norm is not None:
            x = self.ffn_norm(x)

        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class RetBlock(BaseModule):
    def __init__(self,
                 retention: str,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 drop_path=0.,
                 layerscale=False,
                 layer_init_values=1e-5,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                ):
        super(RetBlock, self).__init__(init_cfg)

        self.layerscale = layerscale
        self.embed_dim = embed_dim

        # Normalization Layer
        self.norm1_name, norm1 = build_norm_layer(
                    norm_cfg, self.embed_dim, postfix=2)
        self.add_module(self.norm1_name, norm1)

        self.norm2_name, norm2 = build_norm_layer(
                    norm_cfg, self.embed_dim, postfix=2)
        self.add_module(self.norm2_name, norm2)

        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path))

        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.norm1(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.retention(self.norm1(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x
    
class PatchMerging(BaseModule):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,
                 dim,
                 out_dim,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                 **kwargs):
        super(PatchMerging, self).__init__(init_cfg)

        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        # self.norm = build_norm_layer(norm_cfg, out_dim)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
    
class VisRetNetLayer(BaseModule):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, 
                 embed_dim,
                 out_dim, 
                 depth, 
                 num_heads,
                 init_value: float, 
                 heads_range: float,
                 ffn_dim=96., 
                 drop_path=0., 
                 norm_cfg=dict(type='BN', eps=1e-6),
                 chunkwise_recurrent=False,
                 downsample: PatchMerging = None, 
                 layerscale=False, 
                 layer_init_values=1e-5,
                 init_cfg=None,
                 **kwargs):
        super(VisRetNetLayer, self).__init__(init_cfg)
        self.embed_dim = embed_dim
        self.depth = depth
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag,
                     embed_dim,
                     num_heads,
                     ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, 
                     layerscale, 
                     layer_init_values
                     )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

   
class PatchEmbed(BaseModule):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, 
                 in_chans=3, 
                 embed_dim=96,
                 norm_layer=None,
                 ):
        super(PatchEmbed, self).__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x

@BACKBONES.register_module() 
class VisRetNet(BaseBackbone):

    arch_settings = {
        'tiny':
        dict(
            embed_dims=[64, 128, 256, 512],
            depths=[2, 2, 8, 2],
            num_heads=[4, 4, 8, 16],
            init_values=[2, 2, 2, 2],
            heads_ranges=[4, 4, 6, 6],
            mlp_ratios=[3, 3, 3, 3],
            drop_path_rate=0.1,
            chunkwise_recurrents=[True, True, False, False],
            layerscales=[False, False, False, False]
            ),
        'small':
        dict(
            embed_dims=[64, 128, 256, 512],
            depths=[3, 4, 18, 4],
            num_heads=[4, 4, 8, 16],
            init_values=[2, 2, 2, 2],
            heads_ranges=[4, 4, 6, 6],
            mlp_ratios=[4, 4, 3, 3],
            drop_path_rate=0.15,
            chunkwise_recurrents=[True, True, True, False],
            layerscales=[False, False, False, False]
        ),
        'base':
        dict(
            embed_dims=[80, 160, 320, 512],
            depths=[4, 8, 25, 8],
            num_heads=[5, 5, 10, 16],
            init_values=[2, 2, 2, 2],
            heads_ranges=[5, 5, 6, 6],
            mlp_ratios=[4, 4, 3, 3],
            drop_path_rate=0.4,
            chunkwise_recurrents=[True, True, True, False],
            layerscales=[False, False, True, True],
            layer_init_values=1e-6
        ),
        'large':
        dict(
            embed_dims=[112, 224, 448, 640],
            depths=[4, 8, 25, 8],
            num_heads=[7, 7, 14, 20],
            init_values=[2, 2, 2, 2],
            heads_ranges=[6, 6, 6, 6],
            mlp_ratios=[4, 4, 3, 3],
            drop_path_rate=0.5,
            chunkwise_recurrents=[True, True, True, False],
            layerscales=[False, False, True, True],
            layer_init_values=1e-6
        )    
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3, 
                 out_indices=(3, ),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 patch_norm=True, 
                 layer_init_values=1e-6,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 ):
        super(VisRetNet, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert max(out_indices) < len(arch['depths'])

        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.num_layers = len(self.arch['depths'])
        self.embed_dims = self.arch['embed_dims']
        self.mlp_ratios = self.arch['mlp_ratios']
        self.depths= self.arch['depths']
        self.num_heads = self.arch['num_heads']
        self.init_values = self.arch['init_values']
        self.heads_ranges = self.arch['heads_ranges']
        self.chunkwise_recurrents = self.arch['chunkwise_recurrents']
        self.layerscales = self.arch['layerscales']

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=self.in_channels, embed_dim=self.embed_dims[0])

        # stochastic depth decay rule
        dpr = np.linspace(0, self.arch['drop_path_rate'], sum(self.arch['depths']))

        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = VisRetNetLayer(
                embed_dim=self.embed_dims[i],
                out_dim=self.embed_dims[i + 1] if (i < self.num_layers - 1) else None,
                depth=self.depths[i],
                num_heads=self.num_heads[i],
                init_value=self.init_values[i],
                heads_range=self.heads_ranges[i],
                ffn_dim=int(
                            self.mlp_ratios[i] * self.embed_dims[i]
                            ),
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                chunkwise_recurrents=self.chunkwise_recurrents[i],
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                layerscales=self.layerscales[i],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        
        # print("Begin:", x.shape)
        x = self.patch_embed(x)
        # print("Patch Embed:", x.shape)
        
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)
            # print("Layer {}:".format(i), x.shape)

            if i in self.out_indices:
                outs.append(x)

        
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(VisRetNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
