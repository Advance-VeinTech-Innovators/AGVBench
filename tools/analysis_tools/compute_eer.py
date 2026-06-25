import argparse
import importlib
import os
import os.path as osp
import time
import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from agvbench.datasets import build_dataloader, build_dataset
from agvbench.models import build_model
from agvbench.utils import (get_root_logger, dist_forward_collect, print_log,
                            setup_multi_processes, nondist_forward_collect, traverse_replace)


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Compute EER for Biometric models')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--head', default="head0", help='choose head')
    parser.add_argument('--num_class', type=int, default=None, help='number of classes')
    parser.add_argument('--dataset', default='tju600', help='name of datasets')
    parser.add_argument('--work_dir', default='work_dirs/eer', help='base dir to save results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    setup_multi_processes(cfg)

    # 1. Automatically generate directory structure
    checkpoint_abs = osp.abspath(args.checkpoint)
    path_parts = checkpoint_abs.split(os.sep)

    # Find the position of the dataset name in the path and use it as an anchor to extract subsequent levels
    if args.dataset in path_parts:
        ds_idx = path_parts.index(args.dataset)
        # Extract all levels from the dataset name to the folder containing. pth
        rel_mirrored_path = osp.join(*path_parts[ds_idx:-1])
    else:
        rel_mirrored_path = args.dataset

    # Final save directory = WORK-DIR + rel_mirrored_path
    save_base_dir = osp.join(args.work_dir, rel_mirrored_path)
    mmcv.mkdir_or_exist(osp.abspath(save_base_dir))

    # 2. Determine file name
    epoch_name = osp.splitext(osp.basename(args.checkpoint))[0]  # 如 epoch_600
    log_file = osp.join(save_base_dir, f'eer_{epoch_name}.log')
    npy_dir = osp.join(save_base_dir, epoch_name)
    mmcv.mkdir_or_exist(osp.abspath(npy_dir))

    # Update cfg internal path to prevent MMCV default logic from running randomly
    cfg.work_dir = save_base_dir
    cfg.gpu_ids = [args.gpu_id]
    cfg.model.pretrained = None

    # 3. Initialize Logger
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    print_log(f"Config file: {args.config}", logger=logger)
    print_log(f"Checkpoint: {args.checkpoint}", logger=logger)
    print_log(f"Results will be saved to: {save_base_dir}", logger=logger)

    # 4. Building data and models
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 5. Testing and result saving
    model = MMDataParallel(model, device_ids=[0])
    print_log("Computing EER...", logger=logger)

    outputs = single_gpu_test(model, data_loader)

    if args.head == "head0" and args.head not in outputs.keys():
        args.head = "acc_aug_q"

    # Call the eer method of the dataset and pass npy_dir to save the data
    result, fpr, tpr_list = dataset.eer(
        outputs[args.head],
        num_class=args.num_class,
        work_dir=npy_dir,
        name=epoch_name
    )

    print_log(f"EER Result: {result * 100:.4f}%", logger=logger)
    for i in range(len(tpr_list)):
        print_log(f"FPR@TPR={tpr_list[i]}: {1 - fpr[i]:.6f}", logger=logger)


if __name__ == '__main__':
    main()