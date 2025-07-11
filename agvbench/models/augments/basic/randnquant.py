import torch

@torch.no_grad()
def randnquant(imgs,
               region_num=4,
               collapse_to_val='inside_random',
               spacing='random',
               apply_prob=1.0,
               dist_mode=False,
               **kwargs):
    """
    Randomized Quantization Augmentation

    Args:
        imgs (Tensor): (N, C, H, W)
        region_num (int): Number of value regions to quantize
        collapse_to_val (str): 'middle', 'inside_random', or 'all_zeros'
        spacing (str): 'uniform' or 'random' spacing between regions
        apply_prob (float): Probability to apply quantization

    Returns:
        Tensor: quantized image (N, C, H, W)
    """
    EPSILON = 1
    if not isinstance(imgs, torch.Tensor):
        raise TypeError("imgs must be a torch.Tensor")

    B, C, H, W = imgs.shape
    x = imgs.clone()
    x_ = x.clone()

    flat = x.view(B * C, -1)
    min_val = flat.min(dim=1)[0].view(B, C, 1, 1)
    max_val = flat.max(dim=1)[0].view(B, C, 1, 1)

    # compute region boundaries
    if spacing == "random":
        region_percentiles = torch.rand(B, C, region_num - 1, device=x.device)  # [B, C, R-1]
    elif spacing == "uniform":
        region_percentiles = torch.linspace(0, 1, region_num + 1, device=x.device)[1:-1].view(1, 1, -1).expand(B, C, -1)
    
    region_percentiles = region_percentiles.sort(dim=2)[0]  # [B, C, R-1]
    region_bounds = region_percentiles.unsqueeze(-1) * (max_val - min_val) + min_val  # [B, C, R-1, 1]
    max_val_expanded = (max_val + EPSILON).expand(B, C, 1, 1)  # [B, C, 1, 1]
    region_rights = torch.cat([region_bounds, max_val_expanded], dim=2)  # [B, C, R, 1]
    region_lefts = torch.cat([min_val, region_bounds], dim=2)  # [B, C, R, 1]
    region_mids = (region_rights + region_lefts) / 2  # [B, C, R, 1]

    x_flat = x.unsqueeze(2)  # [B, C, 1, H, W]
    region_mask = (x_flat < region_rights.unsqueeze(-1)) & (x_flat >= region_lefts.unsqueeze(-1))  # [B, C, R, H, W]
    assert (region_mask.sum(2) == 1).all(), "Each pixel must fall in exactly one region"

    region_ids = region_mask.float().argmax(dim=2, keepdim=True)  # [B, C, 1, H, W]

    if collapse_to_val == 'middle':
        proxy = torch.gather(region_mids.unsqueeze(-1).expand(-1, -1, -1, H, W), 2, region_ids)[:, :, 0]  # [B, C, H, W]
    elif collapse_to_val == 'inside_random':
        rand = torch.rand(B, C, region_num, 1, device=x.device)  # [B, C, R, 1]
        region_randoms = region_lefts + rand * (region_rights - region_lefts)  # [B, C, R, 1]
        proxy = torch.gather(region_randoms.unsqueeze(-1).expand(-1, -1, -1, H, W), 2, region_ids)[:, :, 0]  # [B, C, H, W]
    elif collapse_to_val == 'all_zeros':
        proxy = torch.zeros_like(x)  # [B, C, H, W]
    else:
        raise NotImplementedError(f"collapse_to_val={collapse_to_val} not supported")

    out = proxy

    if apply_prob < 1.0:
        mask = (torch.rand(B, 1, 1, 1, device=x.device) < apply_prob).float()
        img = out * mask + x_ * (1 - mask)
    else:
        img = out

    return img
