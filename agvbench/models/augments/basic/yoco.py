import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image, make_grid

@torch.no_grad()
def yoco(img,
         alpha=1.0,
         lam=None,
         dist_mode=False,
         return_mask=False,
         **kwargs):
    r""" YOCO augmentation.

    "You Only Cut Once: Boosting Data Augmentation with a Single 
    Cut. (https://arxiv.org/abs/2201.12078)". In ICML, 2022.
        https://github.com/JunlinHan/YOCO
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """
    augmentation_pool = [
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    ]

    def aug_police(img):
        op1, op2 = random.sample(augmentation_pool, k=2)
        img, img_ = op1(img), op2(img)
        return img, img_


    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        _, _, h, w = img.size()

        img, img_ = aug_police(img)
        if lam > 0.5:
            img = torch.cat((img[:, :, :, 0:int(w/2)], img_[:, :, :, int(w/2):w]), dim=3)
        else:
            img = torch.cat((img[:, :, 0:int(h/2), :], img_[:, :, int(h/2):h, :]), dim=2)

        return img
