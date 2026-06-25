# old implementation
# import torch
# import torch.nn.functional as F
# import numpy as np
#
# @torch.no_grad()
# def randomblur(img,
#                alpha=1.0,
#                lam=None,
#                dist_mode=False,
#                return_mask=False,
#                **kwargs):
#     r""" Random Gaussian Blur Augmentation
#
#     Applies a random blur with kernel size derived from a Beta distribution.
#
#     Args:
#         img (Tensor): Input tensor (N, C, H, W)
#         alpha (float): Beta(alpha, alpha) controls randomness
#         lam (float or None): If None, sample from Beta(alpha, alpha)
#         dist_mode (bool): Reserved for DDP compatibility
#         return_mask (bool): Return blur mask (here, just kernel size per sample)
#     """
#
#     N, C, H, W = img.shape
#     device = img.device
#
#     if lam is None:
#         lam = np.random.beta(alpha, alpha)
#
#     # kernel size proportional to width × lam
#     raw_ks = int(W * lam)
#     kernel_size = raw_ks if raw_ks % 2 == 1 else raw_ks + 1
#     kernel_size = max(3, min(kernel_size, min(H, W) // 2 | 1))  # min cap, odd
#
#     # generate 1D gaussian kernel
#     def get_gaussian_kernel1d(k, sigma):
#         x = torch.arange(k, dtype=torch.float32, device=device) - k // 2
#         kernel = torch.exp(-x**2 / (2 * sigma**2))
#         return kernel / kernel.sum()
#
#     sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
#     k1d = get_gaussian_kernel1d(kernel_size, sigma)
#     kernel2d = k1d[:, None] @ k1d[None, :]  # outer product
#     kernel2d = kernel2d.expand(C, 1, kernel_size, kernel_size)
#
#     padding = kernel_size // 2
#     img = F.conv2d(img, kernel2d, groups=C, padding=padding)
#
#     if return_mask:
#         return img, torch.full((N,), kernel_size, dtype=torch.int)
#
#     return img


# new implementation
import torch
import numpy as np
import torchvision.transforms.functional as F_tv


@torch.no_grad()
def randomblur(img,
               alpha=1.0,
               lam=None,
               max_ratio=0.1,  # [强烈建议保留] 模糊核最大占用图片宽度的比例
               max_kernel=15,  # [强烈建议保留] 模糊核的绝对最大上限
               dist_mode=False,
               return_mask=False,
               **kwargs):
    r""" Random Gaussian Blur Augmentation (Torchvision Version)

    保留了原有的 alpha/lam 接口与 Beta 分布逻辑，
    底层调用官方的 torchvision.transforms.functional.gaussian_blur 实现。
    """
    N, C, H, W = img.shape

    # 1. 保留原版的随机数采样逻辑 (兼容你原有的 MixUp/CutMix 调用体系)
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # 2. 计算 kernel_size (加入安全限制)
    raw_ks = int(W * lam * max_ratio)
    kernel_size = raw_ks if raw_ks % 2 == 1 else raw_ks + 1

    # 限制 kernel_size 最大值，并确保它是 >=3 的奇数
    limit = min(max_kernel, min(H, W) // 2)
    kernel_size = max(3, min(kernel_size, limit | 1))

    # 3. 计算 sigma (这是 OpenCV 和 Torchvision 底层默认的经验公式)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # ================================================================= #
    # 4. 核心替换：使用 torchvision 官方 API，替代手动 F.conv2d
    # ================================================================= #
    # torchvision 的 gaussian_blur 原生支持 (N, C, H, W) 的批量张量处理
    img = F_tv.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    if return_mask:
        # 优化细节：让 mask 张量和 img 保持在同一个设备(CPU/GPU)上
        return img, torch.full((N,), kernel_size, dtype=torch.int32, device=img.device)

    return img