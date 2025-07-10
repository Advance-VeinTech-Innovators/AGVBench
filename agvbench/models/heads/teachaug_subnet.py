import copy
import math
import random
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import BaseModule, force_fp32
from ..registry import HEADS
from .. import builder


def relaxed_bernoulli(logits, temp=0.05, device='cpu'):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits)/temp).sigmoid()


class TriangleWave(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        o = torch.acos(torch.cos(x * math.pi)) / math.pi
        self.save_for_backward(x)
        return o

    @staticmethod
    def backward(self, grad):
        o = self.saved_tensors[0]
        # avoid nan gradient at the peak by replacing it with the right derivative
        o = torch.floor(o) % 2
        grad[o == 1] *= -1 
        return grad


@HEADS.register_module
class TeachAugModule(BaseModule):
    def __init__(self,
                 num_classes=220,
                 scale=1, 
                 hidden=128, 
                 n_dim=128, 
                 dropout_ratio=0.8, 
                 with_context=True,
                 init_cfg=None,
                 **kwargs):
        super(TeachAugModule, self).__init__(init_cfg)
        self.with_context = bool(with_context)
        self.dp = float(dropout_ratio)
        self.n_dim = int(n_dim)
        self.g_aug = AffineTransfer(num_classes, scale=0.5, n_dim=self.n_dim, dropout_ratio=self.dp, with_context=self.with_context)
        self.c_aug = ColorEnhance(num_classes, scale=scale, hidden=hidden, n_dim=self.n_dim, dropout_ratio=self.dp, with_context=self.with_context)

        self.init_weights()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(TeachAugModule, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        # init aug
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def get_params(self, x, y, g_aug, c_aug):
        # sample noise vector from unit gauss
        noise = x.new(x.shape[0], self.n_dim).normal_()
        target = x
        grid = g_aug(target, noise, y)
        scale, shift = c_aug(target, noise, y)
        return grid, scale, shift


    def forward(self, x, y):
        grid, scale, shift = self.get_params(x, y, self.g_aug, self.c_aug)
        # color augmentation
        aug_x = self.c_aug.transform(x, scale, shift)
        # geometric augmentation
        aug_x = self.g_aug.transform(aug_x, grid)

        return aug_x

@HEADS.register_module
class ColorEnhance(BaseModule):
    def __init__(self,
                 n_classes=200,
                 scale=1, 
                 hidden=128, 
                 n_dim=128, 
                 dropout_ratio=0.8, 
                 with_context=True,
                 init_cfg=None,
                 **kwargs):
        super(ColorEnhance, self).__init__(init_cfg)

        self.with_context = with_context
        self.n_classes = n_classes
        self.n_hidden = 4 * n_dim

        # embedding layer for context vector
        if with_context:
            self.context_layer = nn.Conv2d(n_classes, hidden, 1, padding=1//2, bias=False)
        else:
            self.context_layer = None
        # embedding layer for RGB
        self.color_enc1 = nn.Conv2d(3, hidden, 1)
        # body for RGB
        self.color_enc_body = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential()
        )
        # output layer for RGB
        self.c_regress = nn.Conv2d(hidden, 6, 1)
        # body for noise vector
        self.noise_enc = nn.Sequential(
            nn.Linear(n_dim + n_classes if with_context else n_dim, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )
        # output layer for noise vector
        self.n_regress = nn.Linear(self.n_hidden, 2)

        if with_context:
            self.register_parameter('logits', nn.Parameter(torch.zeros(n_classes)))
        else:
            self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset()

        self.with_context = with_context
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.scale = scale
        self.relax = True
        self.stochastic = True

    def sampling(self, scale, shift, y, temp=0.05):
        if self.stochastic: # random apply
            if self.with_context:
                logits = self.logits[y].reshape(-1, 1, 1, 1)
            else:
                logits = self.logits.repeat(scale.shape[0]).reshape(-1, 1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=scale.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            scale = 1 - prob + prob * scale
            shift = prob * shift # omit "+ (1 - prob) * 0"
        return scale, shift

    def forward(self, x, noise, c=None):
        if self.with_context:
            # integer to onehot vector
            onehot_c = nn.functional.one_hot(c, self.n_classes).float()
            noise = torch.cat([onehot_c, noise], 1)
            onehot_c = onehot_c.reshape(*onehot_c.shape, 1, 1)
        gfactor = self.noise_enc(noise)
        gfactor = self.n_regress(gfactor).reshape(-1, 2, 1, 1)
        feature = self.color_enc1(x)
        if self.with_context:
            feature = self.context_layer(onehot_c) + feature
        feature = self.color_enc_body(feature)
        factor = self.c_regress(feature)
        scale, shift = factor.chunk(2, dim=1)
        g_scale, g_shift = gfactor.chunk(2, dim=1)
        scale = (g_scale + scale).sigmoid()
        shift = (g_shift + shift).sigmoid()
        scale = self.scale * (scale - 0.5) + 1
        shift = shift - 0.5
        scale, shift = self.sampling(scale, shift, c)

        return scale, shift

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        nn.init.constant_(self.c_regress.weight, 0)
        nn.init.constant_(self.n_regress.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, scale, shift):
        # ignore zero padding region
        with torch.no_grad():
            h, w = x.shape[-2:]
            mask = (x.sum(1, keepdim=True) == 0).float() # mask pixels having (0, 0, 0) color
            mask = torch.logical_and(mask.sum(-1, keepdim=True) < w,
                                     mask.sum(-2, keepdim=True) < h) # mask zero padding region

        x = (scale * x + shift) * mask
        return TriangleWave.apply(x)

@HEADS.register_module
class AffineTransfer(BaseModule):
    def __init__(self,
                 n_classes=100,
                 scale=0.5,
                 n_dim=128,
                 dropout_ratio=0.2,
                 with_context=True,
                 init_cfg=None,
                 **kwargs):
        super(AffineTransfer, self).__init__(init_cfg)
        hidden = 4 * n_dim

        self.body = nn.Sequential(
            nn.Linear(n_dim + n_classes if with_context else n_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Linear(hidden, 6)
        )
        # identity matrix
        self.register_buffer('i_matrix', torch.Tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3))
        if with_context:
            self.register_parameter('logits', nn.Parameter(torch.zeros(n_classes)))
        else:
            self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset()
        self.with_context = with_context
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.scale = scale
        self.relax = True
        self.stochastic = True

    def sampling(self, x, y=None, temp=0.05):
        if self.stochastic:  # random apply
            if self.with_context:
                logits = self.logits[y].reshape(-1, 1, 1)
            else:
                logits = self.logits.repeat(x.shape[0]).reshape(-1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=logits.device)
            if not self.relax:  # hard sampling
                prob = (prob > 0.5).float()
            return (1 - prob) * self.i_matrix + prob * x
        else:
            return x

    def forward(self, x, noise, c=None):
        if self.with_context:
            with torch.no_grad():
                # integer to onehot vector
                onehot_c = nn.functional.one_hot(c, self.n_classes).float()
                noise = torch.cat([onehot_c, noise], 1)

        A = self.body(noise).reshape(-1, 2, 3)
        A = self.scale * (A.sigmoid() - 0.5) + self.i_matrix
        A = self.sampling(A, c)
        grid = nn.functional.affine_grid(A, x.shape)
        return grid

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        nn.init.constant_(self.logits, 0)

    def transform(self, x, grid):
        x = F.grid_sample(x, grid, mode='bilinear')
        return x
