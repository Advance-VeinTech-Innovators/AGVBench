import torch
import numpy as np


@torch.no_grad()
def softaugment(img,
                t_crop=1.0,
                max_p_crop=1.0,
                pow_crop=2.0,
                bg_crop=1,
                sigma_crop=12,
                iou=False,
                n_classes=220
            ):
    r""" Soft Augmentation.

    "Soft Augmentation for Image Classification. 
    (https://arxiv.org/abs/2211.04625)". In CVPR, 2023.
        https://github.com/youngleox/soft_augmentation
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        t_crop (float): Threshold for overlap ratio to label confidence.
        max_p_crop (float): Maximum probability for the label confidence.
        pow_crop (float): Power for the label confidence mapping.
        bg_crop (float): Background noise strength.
        sigma_crop (int): Standard deviation for random cropping.
        iou (bool): Whether to use IoU for overlap ratio.
        n_classes (int): Number of classes for label confidence mapping.
    Returns:
        img (Tensor): Cropped images with background noise.
        gt_label (Tensor): Soft labels computed from overlap ratio.
    """

    def compute_prob(x, T=0.25, n_classes=220, max_prob=1.0, pow=2.0):
        """Mapping from overlap ratio to label confidence."""
        max_prob = torch.clamp_min(torch.tensor(max_prob), 1 / n_classes)
        T = max(T, 1e-10)

        if x > T:
            return max_prob
        elif x > 0:
            a = (max_prob - 1 / n_classes) / (T ** pow)
            return max_prob - a * (T - x) ** pow
        else:
            return 1 / n_classes

    def draw_offset(sigma=10, limit=24, n=100):
        """Draw integer offset from N(0, sigma^2), clipped to [-limit, limit]."""
        for _ in range(n):
            x = torch.randn(1) * sigma
            if abs(x) <= limit:
                return int(x.item())
        return 0

    N, C, H, W = img.shape

    # Step 1: Create background canvas
    bg = torch.randn((N, C, H * 3, W * 3)).cuda() * bg_crop
    bg[:, :, H : 2 * H, W : 2 * W] = img  # put images in center

    # Step 2: Sample one offset for all
    offset_h = draw_offset(sigma_crop, H)
    offset_w = draw_offset(sigma_crop, W)

    top = offset_h + H
    left = offset_w + W
    bottom = offset_h + 2 * H
    right = offset_w + 2 * W

    img = bg[:, :, top:bottom, left:right]

    # Step 3: compute overlap and soft label
    intersect = (H - abs(offset_h)) * (W - abs(offset_w))
    if iou:
        overlap = intersect / (H * W * 2 - intersect)
    else:
        overlap = intersect / (H * W)

    # We modify the prob of label as lam in our codebase for computing the loss easily
    prob = compute_prob(overlap,
                        T=t_crop,
                        max_prob=max_p_crop,
                        pow=pow_crop,
                        n_classes=n_classes)

    return img, prob

