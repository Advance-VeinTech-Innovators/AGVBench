import argparse
import os
import os.path as osp
import importlib
import torch
import torch.nn as nn
from torchvision import transforms
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
try:
    from skimage.metrics import structural_similarity as ssim
except:
    ssim = None

from agvbench.models.augments.mixups import (
                cutmix, fmix, gridmix, mixup, resizemix, smoothmix,
                augmix, starmix, augmix)
from agvbench.models.augments.basic import yoco
import mmcv
from mmcv import DictAction
from agvbench.utils import get_root_logger, print_log, setup_multi_processes, traverse_replace

def PSNR(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: PSNR value in dB.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def SSIM(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM).

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: SSIM value.
    """
    if img1 is None or img2 is None:
        return 0.0

    # Ensure both images are of the same shape
    if img1.shape != img2.shape:
        return 0.0

    min_dim = min(img1.shape[:2])  # height, width
    if min_dim < 7:
        return 0.0

    win_size = min(11, min_dim)
    if win_size % 2 == 0:
        win_size -= 1

    try:
        # Ensure that channel_axis is set properly
        return ssim(img1, img2, win_size=win_size, multichannel=True, channel_axis=-1)
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0


def apply_gaussian_blur(img, ksize=(5, 5), sigma=0.0):
        """
        Apply Gaussian Blur with parameterized kernel size and sigma.

        Args:
            ksize (tuple): Kernel size as a tuple (height, width).
            sigma (float): Standard deviation for the Gaussian kernel.

        Returns:
            numpy.ndarray: Processed image after applying Gaussian blur.
        """
        if img is not None:
            gaussian_blur_image = cv2.GaussianBlur(
                img, ksize, sigma
            )
        return gaussian_blur_image


def load_sample(path):

    img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    img = transform(img).unsqueeze(0).cuda()

    return img    

def augmentations(img, method, list, value):

    if method in list[0]:
        assert img.shape[0] == 1, "Basic augmentation method just uses one image."
        if method == list[0][0]:
            print(f"Method is: {list[0]}.")
            return apply_gaussian_blur(img, sigma=value)
        elif method == list[0][1]:
            print(f"Method is: {list[1]}.")
            return apply_gaussian_blur(img, sigma=value)
        elif method == list[0][-1]:
            print(f"Method is: {list[2]}.")
            return yoco(img, lam=value)
    elif method in list[-1]:
        assert img.shape[0] <= 1, "Mixup augmentation method just uses more than one image."
        # For Mixup



def parse_args():
    parser = argparse.ArgumentParser(description='Quality Assessment tool of Augmented Image.')
    parser.add_argument('--method', type=str, default='noise',
                        help='the method of augmentation')
    parser.add_argument('--img_path', type=str, default='demo/',
                        help='the path of image for infra testing')
    parser.add_argument('--value', type=float, default=0.5,
                        help='the value of augmented level, from 0-1.')
    parser.add_argument('--work_dir', type=str, default='work_dirs/quality_assessment',
                        help='the path of saved results')
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

    cfg.work_dir = args.work_dir

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger
    log_file = osp.join(cfg.work_dir, 'test_{}_{}.log'.format(args.method, args.value))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)


    aug_list = [['noise', 'blur', 'yoco'],
                ['mixup', 'cutmix', 'smoothmix', 'gridmix', 
                'resizemix', 'augmix', 'starmix', 'fmix']]
    if args.method not in aug_list:
        raise ValueError("Make sure the choose the correct method for augmentation.")
    
    img = load_sample(args.img_path)
    auged_img = augmentations(img, args.method, aug_list, args.value)

    img, auged_img = img.cpu().numpy(), auged_img.cpu().numpy()
    psrn = PSNR(auged_img, img)
    ssim = SSIM(auged_img, img)
    print_log(f"The PSRN of {args.method} method is {psrn}", logger=logger)
    print_log(f"The SSIM of {args.method} method is {ssim}", logger=logger)

if __name__=="__main__":
    main()