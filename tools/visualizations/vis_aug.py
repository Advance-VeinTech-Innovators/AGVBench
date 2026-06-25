import argparse
import os
import torch
import mmcv
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from mmcv import Config
from agvbench.utils import build_from_cfg
from agvbench.models import build_model
from mmcv.runner import load_checkpoint
from agvbench.datasets import PIPELINES
from torchvision.transforms import Compose
from agvbench.models.augments.mixups import (mixup, puzzlemix, guidedmix, starmix, resizemix, gridmix, fmix,
                                             cutmix, saliencymix)
from agvbench.models.augments.basic import cutout, ricap, spnoise, keepaugment, softaugment


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Augmentations')
    parser.add_argument('--img1', help='Path to the first image')
    parser.add_argument('--img2', default=None, help='Path to the second image (optional for mixing)')
    parser.add_argument('--mix', default=False, help='whether to mix')
    parser.add_argument('--config', help='Path to the config file (e.g., r18_mixup.py)')
    parser.add_argument('--checkpoint', help='Path to the model checkpoint file (e.g., r18_mixup.pth)')
    parser.add_argument('--aug_method', default='mixup', help='Augmentation method name')
    parser.add_argument('--save-path', help='Path to save the augmented image')
    return parser.parse_args()


def denormalize(img_tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=img_tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=img_tensor.device)

    if img_tensor.ndim >= 3 and img_tensor.shape[-3] == mean.numel():
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

    img_tensor = img_tensor * std + mean

    if mean.max() <= 1.0:
        img_tensor = img_tensor * 255.0

    return torch.clamp(img_tensor, 0, 255).to(torch.uint8)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    pipeline_cfg = cfg.train_pipeline
    norm_cfg = cfg.img_norm_cfg

    if args.img2 is not None and args.mix:
        print(f"{args.aug_method}......")
        pipeline_cfg.extend([dict(type='ToTensor'), dict(type='Normalize', **norm_cfg)])
        print(pipeline_cfg)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline_cfg]
        pipeline = Compose(pipeline)
        img1 = Image.open(args.img1).convert('RGB')
        img1_tensor = pipeline(img1).unsqueeze(0)
        img2 = Image.open(args.img2).convert('RGB')
        img2_tensor = pipeline(img2).unsqueeze(0)

        batch_imgs = torch.cat([img1_tensor, img2_tensor], dim=0).cuda()
        dummy_labels = torch.LongTensor([0, 1]).cuda()

        print(img1_tensor.max(), img1_tensor.min(), img1_tensor.mean())
        print(img2_tensor.max(), img2_tensor.min(), img2_tensor.mean())

        alpha = cfg.model['alpha']
        if args.aug_method == 'mixup':
            mixed_imgs, _ = mixup(batch_imgs, dummy_labels, alpha=alpha, lam=0.3)
        elif args.aug_method in ["puzzlemix", "guidedmix"]:
            model = build_model(cfg.model)
            if args.checkpoint is not None:
                # Mapping the weights to GPU may cause unexpected video memory leak
                # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
                load_checkpoint(model, args.checkpoint, map_location='cpu')
            model = model.cuda()
            return_mask, mask = False, None
            mix_args_default = model.mix_args[args.aug_method]
            mix_args = cfg.model['mix_args'][args.aug_method]
            mix_args.update(mix_args_default)
            features = model._features(
                batch_imgs, gt_label=dummy_labels, cur_mode=args.aug_method, **mix_args)
            mix_args = dict(alpha=alpha, dist_mode=False, return_mask=return_mask,
                            features=features, **mix_args)
            mixed_imgs, _ = model.dynamic_mode[args.aug_method](batch_imgs, dummy_labels, **mix_args)
        elif args.aug_method == 'starmix':
            mixed_imgs, _ = starmix(batch_imgs, dummy_labels, alpha=alpha, lam=0.7)
        elif args.aug_method == 'ricap':
            mixed_imgs, _ = ricap(batch_imgs, dummy_labels, alpha=alpha, lam=[0.3, 0.3])
        elif args.aug_method == 'cutmix':
            mixed_imgs, _ = cutmix(batch_imgs, dummy_labels, alpha=alpha, lam=0.7)
        elif args.aug_method == 'saliencymix':
            mixed_imgs, _ = saliencymix(batch_imgs, dummy_labels, alpha=alpha, lam=0.7)
        elif args.aug_method in ['resizemix', 'fmix', "gridmix"]:
            model = build_model(cfg.model)
            if args.checkpoint is not None:
                # Mapping the weights to GPU may cause unexpected video memory leak
                # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
                load_checkpoint(model, args.checkpoint, map_location='cpu')
            return_mask, mask = False, None
            mix_args_default = model.mix_args[args.aug_method]
            mix_args = cfg.model['mix_args'][args.aug_method]
            mix_args.update(mix_args_default)
            mix_args = dict(alpha=alpha, dist_mode=False, return_mask=return_mask, **mix_args)
            # lam=0.3
            mixed_imgs, _ = model.static_mode[args.aug_method](batch_imgs, dummy_labels, lam=0.3, **mix_args)

        else:
            raise NotImplementedError

        img1 = denormalize(mixed_imgs[0], **norm_cfg)
        img1 = T.ToPILImage()(img1.cpu())
        img2 = denormalize(mixed_imgs[1], **norm_cfg)
        img2 = T.ToPILImage()(img2.cpu())
        save_paths = args.save_path.split(".")
        img1.save(f"{save_paths[0]}_1.{save_paths[1]}")
        img2.save(f"{save_paths[0]}_2.{save_paths[1]}")

        print(f"aug sample has been saved in: {save_paths[0]}_1.{save_paths[1]}")
        print(f"aug sample has been saved in: {save_paths[0]}_2.{save_paths[1]}")

    else:
        print(f"{args.aug_method}......")
        if args.aug_method in ['translate', 'autoaug', 'trivialaug']:
            print(pipeline_cfg)
            pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline_cfg]
            pipeline = Compose(pipeline)
            img = Image.open(args.img1).convert('RGB')
            img = pipeline(img)
            img.save(args.save_path)

        elif args.aug_method in ['cutout', 'noise', 'keepaug', 'softaugment', "gridmask"]:
            pipeline_cfg.extend([dict(type='ToTensor')])
            print(pipeline_cfg)
            pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline_cfg]
            pipeline = Compose(pipeline)
            img = Image.open(args.img1).convert('RGB')
            img = pipeline(img).unsqueeze(0).cuda()

            model = build_model(cfg.model)
            if args.checkpoint is not None:
                # Mapping the weights to GPU may cause unexpected video memory leak
                # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
                load_checkpoint(model, args.checkpoint, map_location='cpu')
            model = model.cuda()

            if args.aug_method in ["noise"]:
                args.aug_method = 'spnoise'
            if args.aug_method in ["keepaug"]:
                args.aug_method = 'keepaugment'
            aug_args_default = model.aug_args[args.aug_method]
            aug_args = cfg.model['aug_args'][args.aug_method]
            aug_args.update(aug_args_default)

            print("aug_args:", aug_args)
            print(img.max(), img.min(), img.mean())

            return_mask, mask = False, None  # return sample mask in [N, 1, H, W]

            # applying masking sample augmentation methods
            if args.aug_method in ["cutout", "spnoise"]:
                if args.aug_method in ["cutout", "gridmask"]:
                    img = model.masking_mode[args.aug_method](img, alpha=1.0, dist_mode=False,
                                                              return_mask=return_mask, **aug_args)
                elif args.aug_method in ["spnoise"]:
                    img = model.masking_mode[args.aug_method](img, dist_mode=False,
                                                              return_mask=return_mask, **aug_args)
                if return_mask:
                    img, _ = img  # (img, mask): get mask
            elif args.aug_method in ["softaugment", "keepaugment"]:
                if args.aug_method == 'softaugment':
                    img, _ = model.policy_mode[args.aug_method](img, **aug_args)
                elif args.aug_method == 'keepaugment':
                    out = model.backbone(img)[0]
                    if isinstance(out, list):
                        out = out[0]
                    pred_raw = out.clone().detach()
                    # pred_raw = self.backbone(img)[0].clone().detach()
                    dummy_labels = torch.LongTensor([1]).cuda()
                    img = model.policy_mode[args.aug_method](img, dummy_labels, pred_raw, **aug_args)

            img = img.squeeze(0)
            img = T.ToPILImage()(img.cpu())
            img.save(args.save_path)
        else:
            raise NotImplementedError

        print(f"aug sample has been saved in: {args.save_path}")


if __name__ == "__main__":
    main()
