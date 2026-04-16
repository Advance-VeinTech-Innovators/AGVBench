#!/usr/bin/env python3
"""
Simplified ERF Visualization Tool

Usage:
    python vis_erf_simple.py config.py checkpoint.pth --image-path image.jpg
    python vis_erf_simple.py config.py checkpoint.pth --use-random-input
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from mmcv import Config
from mmcv.runner import load_checkpoint
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from agvbench.models import build_model

# Configure matplotlib and seaborn for publication-quality plots (following raw_ref_vis.py style)
plt.rcParams["font.family"] = "Times New Roman"
large = 24
med = 24
small = 24
params = {
    'axes.titlesize': large,
    'legend.fontsize': med,
    'figure.figsize': (16, 10),
    'axes.labelsize': med,
    'xtick.labelsize': med,
    'ytick.labelsize': med,
    'figure.titlesize': large
}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.rc('font', **{'family': 'Times New Roman'})
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple ERF Visualization')
    parser.add_argument('config', help='Model config file')
    parser.add_argument('checkpoint', help='Model checkpoint file')
    parser.add_argument('--image-path', help='Path to input image')
    parser.add_argument('--use-random-input', action='store_true', 
                       help='Use random input instead of image')
    parser.add_argument('--input-size', type=int, default=1024, 
                       help='Input image size')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--save-path', default='erf_heatmap.png', 
                       help='Path to save heatmap')
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    """Load model from config and checkpoint."""
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = build_model(cfg.model)
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        print(f'Loaded checkpoint: {checkpoint_path}')
    
    # Move to device and set eval mode
    model.to(device)
    model.eval()
    
    return model


def prepare_input(args):
    """Prepare input tensor."""
    if args.use_random_input:
        # Generate random input
        input_tensor = torch.randn(1, 3, args.input_size, args.input_size)
        print(f'Generated random input: {input_tensor.shape}')
    
    elif args.image_path:
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        
        image = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        print(f'Loaded image: {args.image_path}, shape: {input_tensor.shape}')
    
    else:
        raise ValueError("Must specify either --image-path or --use-random-input")
    
    return input_tensor


def compute_erf(model, input_tensor, device):
    """Compute Effective Receptive Field."""
    # Move input to device and enable gradients
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    
    # Forward pass
    if hasattr(model, 'forward_backbone'):
        outputs = model.forward_backbone(input_tensor)[-1]
    else:
        outputs = model(input_tensor)
    
    print(f'Model output shape: {outputs[0].shape}')
    
    # Get center point of output feature map
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, input_tensor)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def create_erf_heatmap(data, colormap='RdYlGn', figsize=(10, 10.75), save_path=None):
    """Create ERF heatmap following raw_ref_vis.py style."""
    fig = plt.figure(figsize=figsize, dpi=40)

    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, 
                cmap=colormap,
                center=0, 
                annot=False, 
                cbar=False, 
                annot_kws={"size": 24}, 
                fmt='.2f')
    
    # Add a nicer colorbar on top of the figure (following raw_ref_vis.py)
    try:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mpl_toolkits.axes_grid1.colorbar import colorbar
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('top', size='5%', pad='2%')
        colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    except ImportError:
        # Fallback to standard colorbar if mpl_toolkits is not available
        plt.colorbar(ax.get_children()[0], orientation='horizontal', pad=0.1)
    
    # Adjust layout to minimize margins
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def get_erf_rectangle(data, thresh):
    """Calculate ERF rectangle area (following raw_ref_vis.py logic)."""
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None, None


def visualize_erf(erf_map, save_path):
    """Create and save ERF heatmap using raw_ref_vis.py style."""
    print(f'✓ ERF raw statistics - Min: {np.min(erf_map):.6f}, Max: {np.max(erf_map):.6f}')
    
    # Process data following raw_ref_vis.py approach
    data = np.log10(erf_map + 1)  # Log transform for better readability
    data = data / np.max(data)    # Rescale to [0,1] for comparability among models
    
    print('\n' + '='*60)
    print('ERF High-Contribution Area Analysis (raw_ref_vis.py style)')
    print('='*60)
    
    # Analyze ERF areas for different thresholds (following raw_ref_vis.py)
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        side_length, area_ratio = get_erf_rectangle(data, thresh)
        if side_length is not None:
            print(f'Threshold {thresh:.2f}: Rectangle side = {side_length}, Area ratio = {area_ratio:.6f}')
        else:
            print(f'Threshold {thresh:.2f}: No rectangle found')
    
    # Create heatmap using raw_ref_vis.py style
    create_erf_heatmap(data, colormap='RdYlGn', save_path=save_path)
    
    print(f'✓ ERF heatmap saved: {save_path}')


def main():
    """Main function."""
    args = parse_args()
    
    print("🔥 Simple ERF Visualization Tool (raw_ref_vis.py style)")
    print("="*60)
    
    print("📦 Loading model...")
    model = load_model(args.config, args.checkpoint, args.device)

    print("\n📸 Preparing input...")
    input_tensor = prepare_input(args)

    print("\n🧮 Computing ERF...")
    erf_map = compute_erf(model, input_tensor, args.device)
    
    print(f"\n🎨 Creating visualization...")
    visualize_erf(erf_map, args.save_path)
    
    print(f"\n✅ ERF visualization completed!")
    print(f"📁 Results saved to: {args.save_path}")
    print("🎨 Visualization style: raw_ref_vis.py compatible")


if __name__ == '__main__':
    main()