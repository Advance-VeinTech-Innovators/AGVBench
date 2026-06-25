import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pad_tensor(x, w, H, W):
    if H % w == 0 and W % w == 0:
        return x, (H, W)
    B, C = x.shape[:2]
    if len(x.shape) == 3:
        x = x.view(B, C, H, W)

    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    x = F.pad(x, (0, newW - W, 0, newH - H))

    return x, (newH, newW)


"""PyTorch code for local scan and local reverse"""


def local_scan(x, w=4, H=8, W=8, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, L, C]
    """
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    assert H % w == 0 and W % w == 0
    Hg, Wg = H // w, W // w
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, C)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, C)
    if flip:
        x = x.flip([1])
    return x


def local_scan_bchw(x, w=4, H=8, W=8, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, C, H, W]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, _, _ = x.shape
    x = x.view(B, C, H, W)
    assert H % w == 0 and W % w == 0
    Hg, Wg = H // w, W // w
    if column_first:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
    else:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
    if flip:
        x = x.flip([-1])
    return x


def local_reverse(x, w=4, H=8, W=8, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, L, C]
    """
    B, L, C = x.shape
    assert H % w == 0 and W % w == 0
    Hg, Wg = H // w, W // w

    if flip:
        x = x.flip([-1])

    if column_first:
        x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
    else:
        x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x


class MultiScan(nn.Module):
    DEFAULT = ('h', 'h_flip')
    GLOBAL_DIRECTIONS = ('h', 'h_flip', 'v', 'v_flip')
    LOCAL_DIRECTIONS = ('w2', 'w2_flip', 'w4', 'w4_flip')
    GL_DIRECTIONS = ('h', 'h_flip', 'w2', 'w2_flip')
    GL_DIRECTIONS_INVERSE = ('h_flip', 'v_flip', 'w2_flip', 'w4_flip')
    TOTAL_DIRECTIONS = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w4', 'w4_flip')

    def __init__(self, dim=None, directions=None, seqlens=(8, 8), mode='add'):
        super().__init__()
        assert mode in ['cat', 'add']
        self.seqlens = seqlens
        self.directions = directions
        self.mode = mode
        if mode == "add":
            self.weights = nn.Parameter(torch.ones(len(self.directions), 1, 1, 1))
        else:
            assert dim is not None
            self.dim = dim
            self.proj = nn.Linear(dim*len(self.directions), dim)

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.mode == 'add':
            xs = torch.stack(xs)
            x = (xs * self.weights).sum(0)
        else:
            x = torch.cat(xs, dim=-1)
            x = self.proj(x)
        return x

    def multi_scan(self, x):
        """
        Input x: shape [B, L, D]
        Onput xs: shape [[B, L, D], [B, L, D], ..., [B, L, D]]
        """
        xs = []
        for direction in self.directions:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        """
        Input xs: shape [[B, L, D], [B, L, D], ..., [B, L, D]]
        Onput xs: shape [[B, L, D], [B, L, D], ..., [B, L, D]]
        """
        new_xs = []
        for x, direction in zip(xs, self.directions):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.seqlens
        if len(x.shape) == 3:
            if direction == 'h':
                return x
            elif direction == 'h_flip':
                return x.flip([1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b (w h) d', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b (w h) d', h=H, w=W).flip([1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.seqlens
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([1])
        elif direction == 'v':
            return rearrange(x, 'b (h w) d -> b (w h) d', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([1]), 'b (h w) d -> b (w h) d', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.directions)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')