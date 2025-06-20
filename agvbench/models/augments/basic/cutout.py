import numpy as np
import torch

@torch.no_grad()
def cutout(img,
           alpha=1.0,
           lam=None,
           dist_mode=False,
           return_mask=False,
           **kwargs):
    r""" Cutout augmentation.

    "Improved Regularization of Convolutional Neural Networks with 
    Cutout (https://arxiv.org/abs/1708.04552)". In ICCV, 2019.
        https://github.com/uoguelph-mlrg/Cutout
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def rand_bbox(size, lam, return_mask=False):
        """ generate random box by lam """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        if not return_mask:
            return bbx1, bby1, bbx2, bby2
        else:
            mask = torch.ones((1, 1, W, H)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 0
            mask = mask.expand(size[0], 1, W, H)  # (N, 1, H, W)
            return bbx1, bby1, bbx2, bby2, mask


    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        if len(img.size()) == 5:   # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        _, _, h, w = img.size()

        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam, True)
        img[:, :, bbx1:bbx2, bby1:bby2] = 0
        if return_mask:
            img = (img, mask)

        return img

    # dist mixup with cross gpus shuffle
    else:
        if len(img.size()) == 5:  # self-supervised img [N, 2, C, H, W]
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        _, _, h, w = img.size()

        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam, True)
        img[:, :, bbx1:bbx2, bby1:bby2] = 0
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)

        return img
