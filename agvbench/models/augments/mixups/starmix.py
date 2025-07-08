import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from agvbench.models.utils import batch_shuffle_ddp


@torch.no_grad()
def starmix(img,
            gt_label,
            alpha=1.0,
            lam=None,
            is_vit=False,
            scale=16,
            vis_mask=False,
            dist_mode=False,
            return_mask=False,
            **kwargs):

    def gaussian_kernel(kernel_size, w, h, sigma):

        s = kernel_size * 2
        grid = torch.stack([torch.arange(s).repeat(s).view(s, s), 
                            torch.arange(s).repeat(s).view(s, s).t()], dim=-1
                        ).cuda()
        grid = torch.roll(torch.roll(grid, w, 0), h, 1)
        crop = kernel_size // 2
        grid = grid[crop: s - crop, crop: s - crop]
        
        return torch.exp(-torch.sum((grid - (s - 1) / 2) ** 2, dim=-1) / (2 * sigma ** 2)).view(kernel_size, kernel_size)

    def x_mask(lam, h, w):

        sigma1, sigma2 = lam * h, (1 - lam) * h
        masks = [
                    gaussian_kernel(h, h, w, sigma1), 
                    gaussian_kernel(h, h, w, sigma2), 
                    gaussian_kernel(h, h * 2, w * 2, sigma1)
                ].cuda()

        return (sum(masks) / 3).sigmoid() * lam

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        
        b, _, h, w = img.size()
        y_a, y_b = gt_label, gt_label[rand_index]

        if 0.3 <= lam <= 0.7:
            if is_vit:
                h_, w_ = int(h // scale), int(w // scale)
                mask = x_mask(lam, h_, w_)
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='nearest').squeeze(0).squeeze(0)
            else:
                mask = x_mask(lam, h, w)
            lam = torch.sum(mask) / (h * w)
        else:
            mask = lam

        if vis_mask:
            visualization(mask)
            
        img = img * mask + img_ * (1 - mask)
        if return_mask:
            img = (img, mask.expand(b, 1, h, w))

        return img, (y_a, y_b, lam)

def visualization(mask, name='mask.png'):
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(name)
    plt.show()
    plt.close()

