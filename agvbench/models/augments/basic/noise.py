import numpy as np
import torch

@torch.no_grad()
def spnoise(img,
            prob=0.01,
            noise_type='random',
            dist_mode=False,
            return_mask=False,
            **kwargs):
    r""" Salt & Pepper Noise Augmentation.

    Randomly corrupt pixels in the image with salt (white) or pepper (black)
    noise.

    Args:
        img (Tensor): Input image tensor of shape (N, C, H, W).
        prob (float): Probability of corruption per pixel.
        noise_type (str or None): 'salt', 'pepper', or None for random choice.
        dist_mode (bool): Reserved for DDP-style parallel training.
        return_mask (bool): Whether to return a binary mask indicating noise positions.
    Returns:
        Tensor or (Tensor, Tensor): Noised image, optionally with binary mask.
    """

    if not dist_mode:
        N, C, H, W = img.shape
        device = img.device
        mask = torch.rand((N, 1, H, W), device=device)

        if noise_type == 'random':
            if np.random.rand() < 0.5:
                salt_or_pepper = torch.ones((N, 1, H, W), device=device)
            else:
                salt_or_pepper = torch.zeros((N, 1, H, W), device=device)
        elif noise_type == 'salt':
            salt_or_pepper = torch.ones((N, 1, H, W), device=device)
        elif noise_type == 'pepper':
            salt_or_pepper = torch.zeros((N, 1, H, W), device=device)
        else:
            raise ValueError("noise_type must be None, 'salt', or 'pepper'")

        noise_mask = (mask < prob).expand(-1, C, -1, -1)
        salt_values = salt_or_pepper.expand(-1, C, -1, -1)
        img[noise_mask] = salt_values[noise_mask]

        if return_mask:
            return img, noise_mask.to(torch.uint8)
        
        return img

    else:
        raise NotImplementedError("dist_mode=True not supported in salt_pepper_noise yet.")
