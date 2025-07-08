import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import RandAugment

@torch.no_grad()
def keepaugment(img,
                gt_label,
                model,
                threshold=0.5,
                mode='cut',
                lam=None,
                randaugment_n=2,
                randaugment_m=9
            ):
    r""" Keep Augmentation.

    "KeepAugment: A Simple Information-Preserving Data Augmentation Approach. 
    (https://arxiv.org/abs/2011.11778)". In CVPR, 2021.
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        model (nn.Module): Model to compute confidence scores.
        threshold (float): Threshold for average confidence.
        mode (str): Augmentation mode, either 'cut' or 'paste'.
        lam (float, optional): Lambda value for beta distribution.
        randaugment (RandAugment, optional): RandAugment instance for additional augmentation.
    Returns:
        img (Tensor): Augmented images.
    """

    N, C, H, W = img.shape
    aug = RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)

    if lam is None:
        lam = np.random.beta(1.0, 1.0)

    # Step 1: create mask S (shared)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    region_mask = torch.zeros((1, 1, H, W)).cuda()
    region_mask[:, :, bby1:bby2, bbx1:bbx2] = 1
    region_mask = region_mask.expand(N, 1, H, W)

    # Step 2: forward to get average confidence
    logits = model(img)  # (N, C)
    probs = F.softmax(logits, dim=-1)
    conf = probs[torch.arange(N), gt_label.argmax(dim=-1)]  # (N,)
    avg_conf = conf.mean()

    # Step 3: apply augmentation if condition met
    img_ = img.clone()
    if mode == 'cut':
        if avg_conf < threshold:
            x_aug = x_aug * (1 - region_mask)
    elif mode == 'paste':
        x_prime = img.clone()
        if aug is not None:
            # apply randaugment
            x_prime_list = [aug(T.ToPILImage()(img.cpu())).convert("RGB") for img in x_prime]
            x_prime = torch.stack([T.ToTensor()(img).cuda() for img in x_prime])
        if avg_conf > threshold:
            img = img_ * region_mask + x_prime * (1 - region_mask)

    return img
