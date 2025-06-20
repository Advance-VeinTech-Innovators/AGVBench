import numpy as np
import torch
import torch.nn as nn
import math

from agvbench.models.utils import batch_shuffle_ddp


@torch.no_grad()
def mixpro(img,
           gt_label,
           attn=None,
           alpha=1.0,
           lam=None,
           dist_mode=False,
           num_classes=100,
           smoothing=0.1,
           mask_patch_size=64,
           model_patch_size=16,
           return_mask=False,
           **kwargs):
    r""" MixPro augmentation

    "MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer"
    Basic Image Classification (https://arxiv.org/abs/2304.12043)". In ICLR, 2023.
        https://github.com/fistyee/MixPro
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        attn (Tensor): Attention maps.
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        num_classes (int): Total class of the dataset.
        smoothing (float): Smooth value for the one-hot label.
        mask_patch_size (int): Mask size of raw images.
        model_patch_size (int): Size of each patches.
    """

    def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

    def mixup_target(y_a, y_b, num_classes, lam, smoothing=0.1, device='cuda'):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        y1 = one_hot(y_a, num_classes, on_value=on_value, off_value=off_value, device=device)
        y2 = one_hot(y_b, num_classes, on_value=on_value, off_value=off_value, device=device)

        lam = lam.unsqueeze(1).repeat(1, num_classes)

        return y1 * lam + y2 * (1. - lam)

    def mask_mix(imgs, lam, mask_num, scale_, scale, rand_index):
    
        batch_size = imgs.size(0)

        # total mask_num
        token_count = mask_num ** 2  # 4 * 4 = 16

        # redefine the mask ratios by the tokens
        mask_ratio = [lam for i in range(batch_size)]
        mask_count = [int(np.ceil(token_count * mask_ratio[i])) for i in range(batch_size)]
        mask_ratio = [mask_count[i] / token_count for i in range(batch_size)]

        mask_idx = [np.random.permutation(token_count)[:mask_count[i]] for i in range(batch_size)]
        mask = np.zeros((batch_size, token_count), dtype=int)
        for i in range(batch_size):
            mask[i][mask_idx[i]] = 1
        mask = [mask[i].reshape((mask_num, mask_num)) for i in range(batch_size)]
        mask_ = [mask[i].repeat(scale_, axis=0).repeat(scale_, axis=1) for i in range(batch_size)]  # 64, 64
        mask = [mask[i].repeat(scale, axis=0).repeat(scale, axis=1) for i in range(batch_size)]  # 224, 224
        mask = torch.from_numpy(np.array(mask)).to(imgs.device)
        #  b, 224, 224 -> b, 1, 224, 224 -> b 3 224 224
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)

        mask = mask[:, :, :imgs.shape[2], :imgs.shape[2]]
        img_mix = imgs * mask + imgs[rand_index] * (1 - mask)

        mask_ratio = torch.Tensor(mask_ratio).to(img_mix.device)

        mask_ = np.array(mask_)

        # return samples, targets, mask_, mask_ratio
        return img_mix, mask_, mask_ratio

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    b, _, h, w = img.size()

    mask_num = math.ceil(h / mask_patch_size)  # 224 / 64 = 4
    # num of patch
    scale_ = mask_patch_size // model_patch_size  # 64 / 16 = 4 for ViTs 2 for Swin
    scale = mask_patch_size  # 64
    patch_num = h // model_patch_size  # 224 / 16 = 14 for ViTs and 224 / 32 = 7 for Swin

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        y_a = gt_label
        y_b = gt_label[rand_index]

        # MixPro
        if attn is not None:
            img, mask, lam = mask_mix(img, lam, mask_num, scale_, scale, rand_index)

            mask = torch.from_numpy(mask)
            mask = mask[:, :patch_num, :patch_num]  # 14, 14
            mask = torch.flatten(mask, 1)
            mask = mask.to(img.device)
            if attn.shape[-1] > 49:
                attn = torch.mean(attn[:, :, 0, 1:], dim=1).clone().detach()  # attn from cls_token to images
            else:
                attn = torch.mean(attn[:, :, 0, :], dim=1).clone().detach()
                attn_scale = int(patch_num*patch_num // attn.shape[-1])
                attn = attn.unsqueeze(-1).repeat(1, 1, attn_scale).flatten(start_dim=-2)

            w1, w2 = torch.sum((mask) * attn, dim=1), torch.sum((1 - mask) * attn, dim=1)
            lam_ = w1 / (w1 + w2)

        smooth_label = mixup_target(y_a, y_b, num_classes, lam, smoothing, device='cuda')

        return img, (y_a, y_b, lam), lam_, smooth_label
