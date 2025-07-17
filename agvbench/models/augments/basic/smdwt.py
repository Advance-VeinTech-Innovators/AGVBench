import torch
import pywt
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
# def smdwt_pca(img,
#               thresholds=(0.55, 0.65),
#               wavelet=('bior1.3', 'bior4.4', 'bior6.8'),
#               dist_mode=False, 
#               return_mask=False,
#               **kwargs):
#     r"""
#     Adaptive SMDWT-PCA based augmentation for palm-vein enhancement.

#     Explainable AI: A Multispectral Palm-Vein Identification System
#     with New Augmentation Features (https://dl.acm.org/doi/10.1145/3468873)". In ACM TMCCA, 2019.

#     Args:
#         img (Tensor): Input image tensor of shape (N, 3, 224, 224), assumed in [0, 1] range.
#         thresholds (tuple): (low, high) intensity thresholds for choosing wavelet.
#         wavelet (tuple): Wavelets used for (dark, mid, bright) images.
#         return_mask (bool): If True, return list of wavelets used for each sample.
#     Returns:
#         Tensor: Augmented images, shape (N, 3, 224, 224)
#     """
#     N, C, H, W = img.shape
#     assert C == 3 and H == 224 and W == 224, "Expected shape (N, 3, 224, 224)"
#     device = img.device

#     # Convert RGB to grayscale (0.2989 R + 0.5870 G + 0.1140 B)
#     gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]  # (N, H, W)
#     gray_np = gray.detach().cpu().numpy()  # to numpy for pywt

#     img_ = []
#     mask = []

#     for i in range(N):
#         intensity = gray_np[i].mean()
#         if intensity < thresholds[0]:
#             current_wavelet = wavelet[0]  # dark -> 13/7
#         elif intensity > thresholds[1]:
#             current_wavelet = wavelet[2]  # bright -> 5/3
#         else:
#             current_wavelet = wavelet[1]  # mid -> 9/7

#         mask.append(current_wavelet)

#         # Apply DWT and extract LL
#         coeffs2 = pywt.dwt2(gray_np[i], wavelet=current_wavelet)
#         LL, _ = coeffs2

#         # Convert LL to tensor and interpolate to 224 X 224
#         LL_tensor = torch.tensor(LL, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
#         LL_resized = F.interpolate(LL_tensor, size=(H, W), mode='bicubic', align_corners=False)
#         LL_norm = (LL_resized - LL_resized.min()) / (LL_resized.max() - LL_resized.min() + 1e-8)
#         img_.append(LL_norm)

#     # Stack and repeat channels
#     img = torch.cat(img_, dim=0)  # (N,1,H,W)
#     img = img.repeat(1, 3, 1, 1).to(device)  # (N,3,H,W)

#     if return_mask:
#         return img, mask
#     else:
#         return img

def choose_coff(wavelet='5/3'):
    if wavelet == '5/3':
        coeffs = {
            (-2, -2): -0.03125, (-2, 2): -0.03125, (2, -2): -0.03125, (2, 2): -0.03125,
            (-1, -2): 0.015625, (1, -2): 0.015625, (-2, -1): 0.015625, (-2, 1): 0.015625,
            (-2, 0): 0.0625, (0, -2): 0.0625, (0, 2): 0.0625, (2, 0): 0.0625,
            (-1, -1): 0.09375, (-1, 1): 0.09375, (1, -1): 0.09375, (1, 1): 0.09375,
            (-1, 0): 0.1875, (0, -1): 0.1875, (1, 0): 0.1875, (0, 1): 0.1875,
            (0, 0): 0.5625,
        }
    elif wavelet == '9/7':
        coeffs = {
            (-4, -4): 0.00108, (4, -4): 0.00108, (-4, 4): 0.00108, (4, 4): 0.00108,
            (-3, -4): -0.000683, (-4, -3): -0.000683, (3, -4): -0.000683, (-4, 3): -0.000683,
            (-2, -4): -0.00368, (-4, -2): -0.00368, (2, -4): -0.00368, (-4, 2): -0.00368,
            (-1, -4): 0.01113, (-4, -1): 0.01113, (1, -4): 0.01113, (-4, 1): 0.01113,
            (0, -4): 0.01722, (-4, 0): 0.01722,
            (-3, -3): 0.00043, (3, -3): 0.00043, (-3, 3): 0.00043, (3, 3): 0.00043,
            (-2, -3): 0.00232, (-3, -2): 0.00232, (2, -3): 0.00232, (-3, 2): 0.00232,
            (-1, -3): -0.00702, (-3, -1): -0.00702, (1, -3): -0.00702, (-3, 1): -0.00702,
            (0, -3): -0.01085, (-3, 0): -0.01085,
            (-2, -2): 0.01253, (2, -2): 0.01253, (-2, 2): 0.01253, (2, 2): 0.01253,
            (-1, -2): -0.03786, (-2, -1): -0.03786, (-1, 2): -0.03786, (2, -1): -0.03786,
            (0, -2): -0.05857, (-2, 0): -0.05857,
            (-1, -1): 0.1144, (1, -1): 0.1144, (-1, 1): 0.1144, (1, 1): 0.1144,
            (0, -1): 0.1769, (-1, 0): 0.1769,
            (0, 0): 0.2737
        }
    else:
        raise ValueError("Wrong Wavelet.")
    
    return coeffs
        


def smdwt_pca(img,
              wavelet='5/3',
              dist_mode=False, 
              return_mask=False,
              **kwargs):
    """
    Strict SMDWT-PCA LL-band augmentation using fixed coefficients from the paper.

    Args:
        img (Tensor): Input grayscale images (N, 1, H, W), values in [0,1].
        wavelet (str): '5/3' or '9/7'.
        return_mask (bool): Whether to return coefficient mask positions used.

    Returns:
        Tensor: (N, 1, H, W) augmented LL-band image
        OR (Tensor, List[Dict]): If return_mask=True, also returns per-sample coefficient usage.
    """

    assert img.ndim == 4 and img.shape[1] == 3, "Expected shape (N, 3, H, W)"
    N, _, H, W = img.shape
    device = img.device

    # Convert RGB to grayscale
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]  # (N, H, W)
    gray = gray.unsqueeze(1)  # (N, 1, H, W)

    if wavelet == '5/3':
        kernel_size = 5
        padding = 2

    elif wavelet == '9/7':
        kernel_size = 9
        padding = 4
    else:
        raise ValueError("wavelet must be '5/3' or '9/7'")

    coeffs =  choose_coff(wavelet)

    # unfold patches
    unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=padding)
    patches = unfold(img)  # (N, K*K, H*W)
    K = kernel_size
    output = torch.zeros((N, H * W), device=device)

    for (dy, dx), weight in coeffs.items():
        idx = (K // 2 + dy) * K + (K // 2 + dx)
        output += weight * patches[:, idx]

    out = output.view(N, 1, H, W).clamp(0, 1).repeat(1, 3, 1, 1)


    if return_mask:
        return out, coeffs
    else:
        return out