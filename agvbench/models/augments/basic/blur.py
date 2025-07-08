import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def randomblur(img,
               alpha=1.0,
               lam=None,
               dist_mode=False,
               return_mask=False,
               **kwargs):
    r""" Random Gaussian Blur Augmentation

    Applies a random blur with kernel size derived from a Beta distribution.

    Args:
        img (Tensor): Input tensor (N, C, H, W)
        alpha (float): Beta(alpha, alpha) controls randomness
        lam (float or None): If None, sample from Beta(alpha, alpha)
        dist_mode (bool): Reserved for DDP compatibility
        return_mask (bool): Return blur mask (here, just kernel size per sample)
    """

    N, C, H, W = img.shape
    device = img.device

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # kernel size proportional to width × lam
    raw_ks = int(W * lam)
    kernel_size = raw_ks if raw_ks % 2 == 1 else raw_ks + 1
    kernel_size = max(3, min(kernel_size, min(H, W) // 2 | 1))  # min cap, odd

    # generate 1D gaussian kernel
    def get_gaussian_kernel1d(k, sigma):
        x = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        return kernel / kernel.sum()

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    k1d = get_gaussian_kernel1d(kernel_size, sigma)
    kernel2d = k1d[:, None] @ k1d[None, :]  # outer product
    kernel2d = kernel2d.expand(C, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    img = F.conv2d(img, kernel2d, groups=C, padding=padding)

    if return_mask:
        return img, torch.full((N,), kernel_size, dtype=torch.int)
    
    return img
