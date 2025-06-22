import argparse
import os
import os.path as osp
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from collections import defaultdict

import mmcv
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from openmixup.models import build_model
from openmixup.utils import get_root_logger, print_log, setup_multi_processes, traverse_replace


def calculate_global_sparsity(model):
    """
        Calculate the Global Sparsity
    """
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
    return zero_params / total_params * 100

def calculate_per_layer_sparsity(model, threshold=0.):
    """
        Calculate the Sparsity for per layers
    """
    layer_sparsity = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            total = param.numel()
            if threshold > 0:
                zeros = torch.sum(torch.abs(param) < threshold).item()
            else:
                zeros = torch.sum(param == 0).item()
            layer_sparsity[name] = zeros / total * 100
    return layer_sparsity

def calculate_channel_sparsity(model, threshold=0.):
    """
        Calculate the Sparsity for Channels
    """
    channel_sparsity = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight
            
            out_channels = weight.shape[0]
            channel_norms = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            if threshold > 0:
                zero_channels = torch.sum(channel_norms < threshold).item()
            else:
                zero_channels = torch.sum(channel_norms == 0).item()
            channel_sparsity[name] = zero_channels / out_channels * 100
    return channel_sparsity


def calculate_sparsity_with_threshold(model, threshold=1e-6):
    """
        Calculate the Sparsity with threshold
    """
    total_params = 0
    near_zero_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            near_zero_params += torch.sum(torch.abs(param) < threshold).item()
    
    sparsity = near_zero_params / total_params * 100
    return sparsity


def plot_weight_distribution_all(model, save_path=None, threshold=0.):

    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.detach().cpu().numpy().flatten())
    all_weights = np.concatenate(all_weights)
    print("Total weights are:", len(all_weights))

    counts, bins = np.histogram(all_weights, bins=200)

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid", rc={'grid.linestyle': '--',
                                   "axes.edgecolor": '.20',
                                   })

    # Iterate through each bin, coloring the drawing separately
    for i in range(len(bins) - 1):
        bin_center = 0.5 * (bins[i] + bins[i + 1])
        color = 'lightgreen' if (threshold is not None and abs(bin_center) < threshold) else '#6699cc'
        plt.bar(bin_center, counts[i], width=(bins[i + 1] - bins[i]), color=color, alpha=1.0, log=True)

    plt.title("Global Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_weight_distribution_by_layer(model, save_path=None, layer_keywords=None, max_cols=4):
    """
        A separate subgraph is drawn for each layer, 
        showing the distribution of parameters for all modules (Conv, BN, Linear, etc.) of that layer.
    """
    if layer_keywords is None:
        # Default adjusting for the ResNet, ConvNeXt, Swin and so on
        layer_keywords = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'head']

    layer_weights = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for key in layer_keywords:
            if key in name:
                layer_weights[key].append(param.detach().cpu().numpy().flatten())
                matched = True
                break
        if not matched:
            print("Please make sure the correct module name for print.")
            layer_weights['others'].append(param.detach().cpu().numpy().flatten())

    n_layers = len(layer_weights)
    n_cols = min(n_layers, max_cols)
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, (layer_name, weight_list) in enumerate(layer_weights.items()):
        if not weight_list:
            continue
        weights = np.concatenate(weight_list)
        ax = axes[idx]
        ax.hist(weights, bins=100, color='#6699cc', alpha=1.0, log=True)
        ax.set_title(f"{layer_name}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency (log scale)")
        ax.grid(True)

    # Cleaning the usesless sub-figure
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Per-Layer Weight Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Sparsiry Analysis tool.')
    parser.add_argument('--config', type=str, required=True,
                        help='the path of confiuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='the path of checkpoint file')
    parser.add_argument('--threshold', type=float, default=0.,
                        help='the path of checkpoint file')
    parser.add_argument('--work_dir', type=str, default='work_dirs/sparsity',
                        help='the path of saved results')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='the configuration file`s options')
    
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    setup_multi_processes(cfg)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    work_type = args.checkpoint.split('/')[-1]

    cfg.work_dir = args.work_dir + "/" + work_type
    cfg.gpu_ids = [args.gpu_id]
    cfg.model.pretrained = None
    cfg.threshold = args.threshold
    
    if traverse_replace is not None:
        traverse_replace(cfg, 'memcached', False)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'model_sparsity_thre_{cfg.threshold}_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    print_log(f"Begin calculate the sparsity.", logger=logger)
    
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    
    global_sparsity = calculate_global_sparsity(model)
    layer_sparsity = calculate_per_layer_sparsity(model, threshold=cfg.threshold)
    channel_sparsity = calculate_channel_sparsity(model, threshold=cfg.threshold)
    threshold_sparsity = calculate_sparsity_with_threshold(model, threshold=cfg.threshold)

    print_log(f"Global Sparsity: {global_sparsity:.2f}%", logger=logger)
    print_log(f"Sparsity with each threshold: {threshold_sparsity:.2f}%", logger=logger)
    print_log("Sparsity from each Layers:", logger=logger)
    for name, sparsity in layer_sparsity.items():
        print_log(f"{name}: {sparsity:.2f}%", logger=logger)
    # for name, sparsity in sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True):
        # if sparsity > 0:
            # print_log(f"{name}: {sparsity:.2f}%", logger=logger)
    print_log("Sparsity from each Channels:", logger=logger)
    for name, sparsity in channel_sparsity.items():
        print_log(f"{name}: {sparsity:.2f}%", logger=logger)
        
    # The visualization polt of weight values of model
    plot_weight_distribution_all(
                                 model.module, 
                                 save_path=osp.join(cfg.work_dir, 'weight_distribution_all.svg'),
                                 threshold=cfg.threshold 
                                )
    plot_weight_distribution_by_layer(
                                      model.module,
                                      save_path=osp.join(cfg.work_dir, 'weight_distribution_per_layer.svg'),
                                      layer_keywords=['backbone.layer1', 
                                                      'backbone.layer2',
                                                      'backbone.layer3',
                                                      'backbone.layer4', 
                                                      'head']
                                    #   layer_keywords=['backbone.stem', 
                                    #                   'backbone.stages.0', 
                                    #                   'backbone.stages.1', 
                                    #                   'backbone.stages.2',
                                    #                   'backbone.stages.3',
                                    #                   'backbone.blockneck.0',
                                    #                   'backbone.blockneck.1',
                                    #                   'backbone.blockneck.2', 
                                    #                   'head']
                                    )

    print_log(f"The logs results saved in {log_file}", logger=logger)
    print_log(f"The plot of weight values's distribution are saved in {cfg.work_dir}", logger=logger)


if __name__ == '__main__':
    main() 
