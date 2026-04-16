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

from torchvision.utils import save_image

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
        self.p_aug = PolicyNetwork(n_subpolicies=subpolicies, threshold=threshold )
        self.n_aug = NeuralAugmenter(in_channels=in_channels, noise_std=noise_std, num_classes=num_classes)

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
                 noise_std=0.001,
                 alpha=0.9,
                 num_classes=220,
                 init_cfg=None,
                 **kwargs):
        super(NeuralAugmenter, self).__init__(init_cfg)
        self.noise_std = noise_std
        self.alpha = alpha
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.label_embed = nn.Embedding(num_classes, 64) 
        self.scale = nn.Conv2d(2, 6, kernel_size=1)
        # self.body = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=1),
        #     nn.Conv2d(128, 256, kernel_size=1),
        #     nn.Conv2d(256, 256, kernel_size=1),
        #     nn.Conv2d(256, 2, kernel_size=1),
        # )
        # self.noise = nn.Sequential(
        #     nn.Conv2d(num_classes + 64, 896, kernel_size=1),
        #     nn.Conv2d(896, 1024, kernel_size=1),
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        # )
        # self.neck1 = nn.Conv2d(1024, 2, kernel_size=1)
        # self.neck2 = nn.Conv2d(1024, 6, kernel_size=1)
        self.body = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        self.noise = nn.Sequential(
            nn.Conv2d(in_channels + 64, 128, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1),
        )
        self.neck1 = nn.Conv2d(64, 2, kernel_size=1)
        self.neck2 = nn.Conv2d(64, 6, kernel_size=1)

        self.output = nn.Conv2d(6, in_channels, kernel_size=1)


    def forward(self, x, y):
        # image aug
        x_ = self.conv(x) 
        label_vec = self.label_embed(y).unsqueeze(-1).unsqueeze(-1).expand_as(x_)
        x_ = x_ + label_vec
        x_ = self.body(x_)
        # noise aug
        noise = torch.randn_like(x) * self.noise_std
        noise = torch.cat([noise, label_vec], dim=1)
        x_n = self.noise(noise)
        # output aug
        x_ += self.neck1(x_n)
        x_ = self.scale(x_) + self.neck2(x_n)
        x_ = self.output(x_)

        x = self.alpha * x + (1 - self.alpha) * torch.sigmoid(x_)

        return x

@HEADS.register_module
class PolicyNetwork(BaseModule):
    def __init__(self, 
                 n_subpolicies=10,
                 threshold=0.2,
                 init_cfg=None,
                 **kwargs):
        super(PolicyNetwork, self).__init__(init_cfg)
        self.p = nn.Parameter(torch.rand(n_subpolicies, 2))
        self.m = nn.Parameter(torch.rand(n_subpolicies, 2))
        self.t = threshold
        self.operations = [ rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, 
                           contrast, brightness, sharpness, autocontrast, equalize, posterize,
                    ]

    def forward(self, x, y, backbone, head, temperature=0.1):

        subpolicy_idx = torch.multinomial(self.p.mean(dim=1), 1)
        p_sub = self.p[subpolicy_idx].squeeze(0)
        m_sub = self.m[subpolicy_idx].squeeze(0)

        alpha1 = self._gumbel_softmax(p_sub[0], temperature)
        alpha2 = self._gumbel_softmax(p_sub[1], temperature)

        op1 = np.random.choice(self.operations)
        op2= np.random.choice(self.operations)

        x1 = (alpha1 * self.apply_augment(x, op1, m_sub[0]) + (1 - alpha1) * x) + 1e-4 * m_sub[0]
        x2 = (alpha2 * self.apply_augment(x1, op2, m_sub[1]) + (1 - alpha2) * x1) + 1e-4 * m_sub[1]
        
        if np.random.random() > 0.5:
            saliency_map = self._compute_saliency(x, y, backbone, head)
            if saliency_map.sum() > self.t:
                saliency_map = saliency_map.unsqueeze(1)
                x = x * saliency_map + x2 * (1 - saliency_map)
        return x2

    def _gumbel_softmax(self, p, temp):
        u = torch.rand_like(p)
        l = torch.log(p) - torch.log(1-p) + torch.log(u) - torch.log(1-u)
        return 1 / (1 + torch.exp(-l / temp))
    
    def _compute_saliency(self, x, y, backbone, head):
        x.requires_grad_()
        pred_out = backbone(x)[-1]
        pred_out = head([pred_out])[0]
        scores = torch.gather(pred_out, 1, y.unsqueeze(1)).squeeze(1)
        grad = torch.autograd.grad(scores, x, grad_outputs=torch.ones_like(scores))[0]
        saliency_map = grad.abs().sum(dim=1)
        return saliency_map
    
    def apply_augment(self, images, op, severity):
        augmented_images = []
        for image in images:
            image = np.clip(image.detach().cpu().numpy() * 255., 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(image.transpose(1, 2, 0))  # Convert to PIL.Image
            pil_img = op(pil_img, severity.item())
            augmented_image = np.asarray(pil_img).transpose(2, 0, 1) / 255.
            augmented_images.append(torch.tensor(augmented_image, dtype=torch.float32, device=images.device))
        return torch.stack(augmented_images)