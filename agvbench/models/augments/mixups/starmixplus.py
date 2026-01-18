import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def generate_smooth_mask(_x: int, _y: int, height: int, width: int, 
                        sigma: float = 9.0, lam: float = 0.0, device: torch.device = None, auto_scale: bool = True) -> torch.Tensor:
    """Generate a smooth Gaussian mask for a specified center point
    
    Args:
        _x: Center point x coordinate
        _y: Center point y coordinate  
        height: Image height
        width: Image width
        sigma: Standard deviation of Gaussian kernel
        lam: Mixing ratio parameter
        device: Device type
        auto_scale: Whether to auto-scale sigma based on image size
    
    Returns:
        Smooth mask tensor [H, W]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optionally auto-scale sigma based on image size for better adaptation
    adaptive_sigma = sigma
    if auto_scale:
        # Base sigma is designed for 32x32 images, scale it proportionally
        base_size = 32
        scale_factor = min(height, width) / base_size
        adaptive_sigma = sigma * scale_factor
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Calculate the distance to the center point
    distance_squared = (x_coords - _x) ** 2 + (y_coords - _y) ** 2
    return torch.exp(-distance_squared * lam / (adaptive_sigma ** 2))


def select_top_k_points(features: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the Top-K gradient points with the highest saliency"""
    # features shape: [B, H, W] (from _features method)
    if features.dim() == 3:
        batch_size, height, width = features.shape
        gradient_magnitude = features  # Already [B, H, W]
    elif features.dim() == 4:
        # If it is a 4D tensor, calculate the L2 norm of the channel dimension
        batch_size, _, height, width = features.shape
        gradient_magnitude = torch.norm(features, dim=1)  # [B, H, W]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {features.dim()}D")
    
    device = features.device
    
    # Get the indices of the Top-K points
    flat_gradients = gradient_magnitude.view(batch_size, -1)
    top_k_values, top_k_indices = torch.topk(flat_gradients, k=k, dim=1)
    
    # Convert the indices to 2D coordinates
    top_k_coords = torch.zeros(batch_size, k, 2, device=device, dtype=torch.long)
    top_k_coords[:, :, 0] = top_k_indices % width      # x coordinates
    top_k_coords[:, :, 1] = torch.div(top_k_indices, width, rounding_mode='trunc')  # y coordinates
    
    return top_k_coords, top_k_values



@torch.no_grad()
def starmixplus(img: torch.Tensor,
                gt_label: torch.Tensor,
                features: torch.Tensor,
                alpha=1.0,
                lam=None,
                k=4,
                sigma=30.0,
                dist_mode: bool = False,
                auto_scale_sigma: bool = True,
                **kwargs) -> Tuple[torch.Tensor, Tuple]:
    """PointMix: Point-level mixup based on gradient saliency
    
    Args:
        img: Input image tensor [N, C, H, W]
        gt_label: Ground-truth label tensor
        features: Gradient feature tensor [N, H, W], obtained through the _features method of the training framework
        alpha: Beta distribution parameter
        lam: Mixing ratio
        k: Number of selected top-K gradient points
        sigma: Standard deviation of Gaussian kernel, controlling the smoothness of the mask
        dist_mode: Whether to use distributed mode
        auto_scale_sigma: Whether to automatically scale sigma based on image size (default: True)
        **kwargs: Other parameters

    Returns:
        mixed_img: Mixed image
        (y_a, y_b, lam): Label and mixing ratio
    """

    def create_point_masks(features: torch.Tensor, k: int = 4, sigma: float = 9.0, lam: float = 0.0, auto_scale: bool = True) -> torch.Tensor:
        """Create a set of masks based on the Top-K gradient points
        
        Args:
            features: Gradient feature tensor [B, H, W] (from _features method)
            k: Number of selected points
            sigma: Standard deviation of Gaussian kernel
            lam: Mixing ratio parameter
            auto_scale: Whether to auto-scale sigma based on image size
        
        Returns:
            Normalized mask [B, 1, H, W]
        """
        # features shape: [B, H, W] (from _features method)
        if features.dim() == 3:
            batch_size, height, width = features.shape
        elif features.dim() == 4:
            batch_size, _, height, width = features.shape
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {features.dim()}D")
        
        device = features.device
        
        # Get the Top-K gradient points
        top_k_coords, _ = select_top_k_points(features, k)
        
        # Generate a smooth mask for each point
        batch_masks = []
        for b in range(batch_size):
            point_masks = []
            for i in range(k):
                _x = top_k_coords[b, i, 0].item()
                _y = top_k_coords[b, i, 1].item()
                point_mask = generate_smooth_mask(_x, _y, height, width, sigma, lam, device, auto_scale)
                point_masks.append(point_mask)
            
            # Sum all the masks
            combined_mask = torch.stack(point_masks).sum(dim=0)
            if combined_mask.max() > 0:
                combined_mask = combined_mask / combined_mask.max()
            
            batch_masks.append(combined_mask)
        
        return torch.stack(batch_masks).unsqueeze(1)  # [B, 1, H, W]


    if lam is None:
        lam = np.random.beta(alpha, alpha)
    
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:  # [N, 2, C, H, W]
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a, y_b = gt_label, gt_label[rand_index]
    
        mask = create_point_masks(features, k=k, sigma=sigma, lam=lam, auto_scale=auto_scale_sigma)
    
    if mask.shape[-2:] != img.shape[-2:]:
        mask = F.interpolate(mask, size=img.shape[-2:], mode='bilinear', align_corners=False)
    
    img = mask * img + (1 - mask) * img_
    
    if mask.shape[1] == 1:
        lam = mask.squeeze(1).reshape(mask.shape[0], -1).mean(-1)
    else:
        lam = mask.mean(dim=[2, 3]).mean(dim=1)
    
    return img, (y_a, y_b, lam)


def visualization(mask, name='mask.png'):
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mask[0,:,:,:].squeeze(0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(name)
    plt.show()
    plt.close()

