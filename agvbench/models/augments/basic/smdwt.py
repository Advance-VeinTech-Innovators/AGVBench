import torch
import pywt
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def smdwt_pca(img,
              thresholds=(0.55, 0.65),
              wavelet=('bior1.3', 'bior4.4', 'bior6.8'),
              dist_mode=False, 
              return_mask=False,
              **kwargs):
    r"""
    Adaptive SMDWT-PCA based augmentation for palm-vein enhancement.

    Explainable AI: A Multispectral Palm-Vein Identification System
    with New Augmentation Features (https://dl.acm.org/doi/10.1145/3468873)". In ACM TMCCA, 2019.

    Args:
        img (Tensor): Input image tensor of shape (N, 3, 224, 224), assumed in [0, 1] range.
        thresholds (tuple): (low, high) intensity thresholds for choosing wavelet.
        wavelet (tuple): Wavelets used for (dark, mid, bright) images.
        return_mask (bool): If True, return list of wavelets used for each sample.
    Returns:
        Tensor: Augmented images, shape (N, 3, 224, 224)
    """
    N, C, H, W = img.shape
    assert C == 3 and H == 224 and W == 224, "Expected shape (N, 3, 224, 224)"
    device = img.device

    # Convert RGB to grayscale (0.2989 R + 0.5870 G + 0.1140 B)
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]  # (N, H, W)
    gray_np = gray.detach().cpu().numpy()  # to numpy for pywt

    img_ = []
    mask = []

    for i in range(N):
        intensity = gray_np[i].mean()
        if intensity < thresholds[0]:
            current_wavelet = wavelet[0]  # dark -> 13/7
        elif intensity > thresholds[1]:
            current_wavelet = wavelet[2]  # bright -> 5/3
        else:
            current_wavelet = wavelet[1]  # mid -> 9/7

        mask.append(current_wavelet)

        # Apply DWT and extract LL
        coeffs2 = pywt.dwt2(gray_np[i], wavelet=current_wavelet)
        LL, _ = coeffs2

        # Convert LL to tensor and interpolate to 224 X 224
        LL_tensor = torch.tensor(LL, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        LL_resized = F.interpolate(LL_tensor, size=(H, W), mode='bicubic', align_corners=False)
        LL_norm = (LL_resized - LL_resized.min()) / (LL_resized.max() - LL_resized.min() + 1e-8)
        img_.append(LL_norm)

    # Stack and repeat channels
    img = torch.cat(img_, dim=0)  # (N,1,H,W)
    img = img.repeat(1, 3, 1, 1).to(device)  # (N,3,H,W)

    if return_mask:
        return img, mask
    else:
        return img
