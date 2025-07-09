import argparse
import os
import os.path as osp
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import mmcv
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from agvbench.datasets import build_dataloader, build_dataset
from agvbench.models import build_model
from agvbench.utils import (get_root_logger, dist_forward_collect, print_log,
                             setup_multi_processes, nondist_forward_collect, traverse_replace,)


# For Infra
def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


# For Purning
def prune_by_threshold(model, threshold):
    for name, param in model.named_parameters():
        if param.requires_grad:
            tensor = param.data
            mask = tensor.abs() >= threshold
            tensor.mul_(mask)  # set to zero
    return model

def prune_by_percentage(model, percentage):
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.cpu().abs().flatten())
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, percentage)
    print(f"=> computed threshold from {percentage*100:.2f}% percentile: {threshold:.4e}")
    return prune_by_threshold(model, threshold.item())

def load_sample(path):

    img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    img = transform(img).unsqueeze(0).cuda()

    return img

def parse_args():
    parser = argparse.ArgumentParser(description='Purning Model tool.')
    parser.add_argument('--config', type=str, required=True,
                        help='the path of confiuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='the path of checkpoint file')
    parser.add_argument('--method', type=str, default='threshold',
                        help='the method of purning')
    parser.add_argument('--value', type=float, default=0.,
                        help='the value of purned model')
    parser.add_argument('--demo_path', type=str, default='demo/',
                        help='the path of demo for infra testing')
    parser.add_argument('--work_dir', type=str, default='work_dirs/sparsity/purned_ckpt',
                        help='the path of saved results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                        default='none', help='job launcher')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--local_rank', help='set local_rank for torch.distributed.launch (torch<2.0.0)',
                        type=int, default=0)
    parser.add_argument('--port', type=int, default=29500, help='port only works when launcher=="slurm"')
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

    work_type = args.checkpoint.split('/')[-2]

    cfg.work_dir = args.work_dir
    cfg.gpu_ids = [args.gpu_id]
    cfg.model.pretrained = None

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'test_{}_purned.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.cuda()

    if args.method == 'threshold':
        prune_by_threshold(model, args.value)
    elif args.method == 'percentage':
        prune_by_percentage(model, args.value)
    else:
        raise ValueError("Unsupported prune method")


    if not distributed:
        print_log("Purned Model Classification Results.", logger=logger)
        outputs = single_gpu_test(model, data_loader)

        rank, _ = get_dist_info()
        if rank == 0:
            for name, val in outputs.items():
                dataset.evaluate(torch.from_numpy(val), name, logger, topk=(1, 5))
    

    # Computing the infra time.
    infra_sample = load_sample(args.demo_path)
    start_time = time.time()
    infra_output = model(infra_sample, mode='inference')
    end_time = time.time()
    inference_time = end_time - start_time
    print_log(f"Infra Time: {inference_time * 1000:.2f} ms", logger=logger)
        

    # Saving Purned Model as new checkpoint
    torch.save({'state_dict': model.state_dict()}, 
                '{}/{}_pruned_{}.pth'.format(cfg.work_dir, work_type, args.method)
            )
    print_log(f"Saved pruned model to: {cfg.work_dir}/{work_type}_pruned_{args.method}.pth", logger=logger)

if __name__ == '__main__':
    main() 


