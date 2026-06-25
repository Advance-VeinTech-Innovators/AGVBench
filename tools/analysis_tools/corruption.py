import argparse
import importlib
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
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


def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data', type=str, default=None,
                        help='data file path')
    parser.add_argument('--meta', type=str, default=None,
                        help='meta file path')
    parser.add_argument('--config', type=str, default=None,
                        help='test config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/courrption',
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--local_rank',
        help='set local_rank for torch.distributed.launch (torch<2.0.0)',
        type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    cfg.gpu_ids = [args.gpu_id]

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

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
    dataset_name = args.data.split('/')[-1]
    config = args.config.split('.')[0]
    config_name = (config.split('/')[-1]).split('_')
    model_name = config_name[0]
    if model_name in ["starlknet", "vit", "swin"]:
        model_name = model_name + "_" + config_name[1]
        model_aug_name = model_name + "_" + config_name[2]
    else:
        model_aug_name = model_name + "_" + config_name[1]
    save_base_dir = osp.join(cfg.work_dir, dataset_name, model_name, model_aug_name)
    print('---------------------------------')
    print(save_base_dir)
    print('---------------------------------')
    log_file = osp.join(save_base_dir, f'test_courrption.log')
    if osp.exists(log_file):
        print(f"{log_file} already exists")
        return
    mmcv.mkdir_or_exist(osp.abspath(save_base_dir))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    data_cfg = cfg.data.val
    data_cfg.data_source.list_file = args.meta
    data_cfg.data_source.root = args.data
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    print_log(f"It`s {args.data.split('/')[-1]} test experiment!", logger=logger)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        for name, val in outputs.items():
            dataset.evaluate(
                torch.from_numpy(val), name, logger, topk=(1, 5))


if __name__ == '__main__':
    main()
