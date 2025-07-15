import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import BaseModule
from ..registry import HEADS
from .. import builder
from agvbench.models.utils.augmentation import (autocontrast, equalize, posterize, rotate, solarize,
                                                shear_x, shear_y, translate_x, translate_y, color,
                                                contrast, brightness, sharpness)
from PIL import Image

@HEADS.register_module
class MAdAugBlock(BaseModule):
    def __init__(self,
                 num_classes=220,
                 noise_std=0.1,
                 in_channels=3,
                 subpolicies=10,
                 threshold=0.2,
                 init_cfg=None,
                 **kwargs):
        super(MAdAugBlock, self).__init__(init_cfg)
        self.p_aug = PolicyNetwork(in_channels=in_channels, noise_std=noise_std, num_classes=num_classes)
        self.n_aug = NeuralAugmenter(n_subpolicies=subpolicies, threshold=threshold)

        self.init_weights()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(MAdAugBlock, self).init_weights()
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

    def forward(self, x, y, backbone, head):
        x = self.p_aug(x, y, backbone, head)
        x = self.n_aug(x, y)
        return x

@HEADS.register_module
class NeuralAugmenter(BaseModule):
    def __init__(self,
                 in_channels=3,
                 noise_std=0.1,
                 num_classes=220,
                 init_cfg=None,
                 **kwargs):
        super(NeuralAugmenter, self).__init__(init_cfg)
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.label_embed = nn.Embedding(num_classes, 64) 
        self.noise_fc = nn.Linear(128, 64) 
        self.body = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=1)
        )
        self.noise_std = noise_std

    def forward(self, x, y):

        x = self.conv(x) 
        label_vec = self.label_embed(y).unsqueeze(-1).unsqueeze(-1)
        x = x + label_vec

        noise = torch.randn_like(x) * self.noise_std
        noise = torch.cat([noise, label_vec.expand_as(x)], dim=1)
        noise = self.noise_fc(noise.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_aug = x + noise
        return torch.sigmoid(self.body(x_aug))

@HEADS.register_module
class PolicyNetwork(BaseModule):
    def __init__(self, 
                 backbone,
                 head,
                 n_subpolicies=10,
                 threshold=0.2,
                 init_cfg=None,
                 **kwargs):
        super(PolicyNetwork, self).__init__(init_cfg)
        self.p = nn.Parameter(torch.rand(n_subpolicies, 2))  # [subpolicy, operation]
        self.m = nn.Parameter(torch.rand(n_subpolicies, 2))  # 幅度 \in [0, 1]
        self.t = threshold
        self.operations = [
                        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
                        translate_x, translate_y, color, contrast, brightness, sharpness
                    ]
        self.backbone = backbone
        self.head = head

    def forward(self, x, y, backbone, head, temperature=0.1):

        subpolicy_idx = torch.multinomial(self.p.mean(dim=1), 1)
        p_sub = self.p[subpolicy_idx]
        m_sub = self.m[subpolicy_idx]


        alpha1 = self._gumbel_softmax(p_sub[0], temperature)
        alpha2 = self._gumbel_softmax(p_sub[1], temperature)
        
        x1 = alpha1 * self.apply_augment(x, self.operations[0], m_sub[0]) + (1 - alpha1) * x
        x2 = alpha2 * self.apply_augment(x1, self.operations[1], m_sub[1]) + (1 - alpha2) * x1
        
        if np.random.random() > 0.5:
            saliency_map = self._compute_saliency(x, y, backbone, head)
            if saliency_map.sum() > self.t:
                x = x * saliency_map + x2 * (1 - saliency_map)
        return x2

    def _gumbel_softmax(self, p, temp):
        u = torch.rand_like(p)
        l = torch.log(p) - torch.log(1-p) + torch.log(u) - torch.log(1-u)
        return 1 / (1 + torch.exp(-l / temp))
    
    def _compute_saliency(self, x, y, backbone, head):
        x.requires_grad_()
        scores = backbone(x)
        scores = head(x[-1])[: y]
        grad = torch.autograd.grad(scores, x)[0]
        return grad.abs().sum(dim=1)
    
    def apply_augment(image, op, severity):
        image = np.clip(image * 255., 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image.transpose(1, 2, 0))  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img).transpose(2, 0, 1) / 255.
