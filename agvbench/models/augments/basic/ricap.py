import numpy as np
import torch

@torch.no_grad()
def ricap(img,
          gt_label,
          alpha=1.0,
          lam=None,
          choose_num=2,
          dist_mode=False,
          return_mask=False,
          **kwargs):
    r""" RICAP augmentation.

    "RICAP: Data Augmentation using Random Image Cropping 
    and Patching for Deep CNNs (https://arxiv.org/abs/1811.09030)". In IEEE TCSVT, 2019.
        https://github.com/jackryo/ricap
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        choose_num (int): The number of choosen for cutting and mixing.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    if lam is None:
        lam = np.random.beta(alpha, alpha, choose_num)
    
    img1, img2 = img.size()[choose_num:]

    # generate boundary position (w, h)
    w = int(np.round(img1 * lam[0]))
    h = int(np.round(img2 * lam[-1]))
    w_ = [w, img1 - w, w, img1 - w]
    h_ = [h, h, img2 - h, img2 - h]

    # select four img
    cropped_images = {}
    # gt_label_ = {}
    # lam_ = {}
    gt_label_ = []
    lam_ = []
    for k in range(4):
        index = torch.randperm(img.size(0)).cuda()
        x_k = np.random.randint(0, img1 - w_[k] + 1)
        y_k = np.random.randint(0, img2 - h_[k] + 1)
        cropped_images[k] = img[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        gt_label_.append(gt_label[index])
        lam_.append( (w_[k] * h_[k]) / (img1 * img2) )
    # patch cropped images
    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

    return patched_images, (gt_label_, lam_)

