import numpy as np
import os.path as osp
import pickle
import shutil
import tempfile
import time
import math
import random
from typing import Optional
from torch.autograd import Variable
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms
from torchvision.utils import save_image

import mmcv
from mmcv.runner import get_dist_info
from .gather import gather_tensors_batch


def nondist_forward_collect(func, data_loader, length):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


def dist_forward_collect(func, data_loader, rank, length, ret_rank=-1):
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        rank (int): This process id.
        length (int): Expected length of output arrays.
        ret_rank (int): The process that returns.
            Other processes will return None.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)  # dict{key: tensor}
        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_cat = np.concatenate([batch[k].numpy() for batch in results],
                                     axis=0)
        if ret_rank == -1:
            results_gathered = gather_tensors_batch(results_cat, part_size=20)
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
        else:
            results_gathered = gather_tensors_batch(
                results_cat, part_size=20, ret_rank=ret_rank)
            if rank == ret_rank:
                results_strip = np.concatenate(
                    results_gathered, axis=0)[:length]
            else:
                results_strip = None
        results_all[k] = results_strip
    return results_all


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmcv.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None


def occlusion_forward_collect(func, data_loader, length, drop_ratio, drop_size):

    # create a mask for occlusion test
    patch = 224 // drop_size
    patch_num = patch * patch
    mask_num = round(patch_num * drop_ratio * 0.1) # need mask number

    print(f"patch size is {patch} with the total tokens {patch_num}")
    print(f"occlusion ratio is {drop_ratio * 100 * 0.1}% and masked tokens are {mask_num}")

    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        img = rearrange(data['img'], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=drop_size, p2=drop_size)
        row = np.random.choice(range(patch_num), size=mask_num, replace=False)
        img[:, row, :] = 0.0
        img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=patch, w=patch, p1=drop_size, p2=drop_size)

        data['img'] = img
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


r""" 
    FGSM Adversarial Attack
"""
def fgsm_nondist_forward_collect(func, data_loader, length, head, dataset='vera220'):

    eps = 8
    if dataset == 'vera220':
        mean=[0.4399, 0.4399, 0.4399]
        std=[0.114, 0.114, 0.114]
    elif dataset == 'tju600':
        mean=[0.382, 0.382, 0.382]
        std=[0.088, 0.088, 0.088]
    elif dataset == 'hkpu500':
        mean=[0.556, 0.556, 0.556]
        std=[0.047, 0.047, 0.047]
    elif dataset == 'casia200':
        mean=[0.471, 0.471, 0.471]
        std=[0.067, 0.067, 0.067]
    elif dataset == 'scut1100':
        mean=[0.288, 0.288, 0.288]
        std=[0.264, 0.264, 0.264]
    else: 
        raise ValueError("please chose a valid dataset")
    
    device = img.device
    mean, std = torch.tensor(mean).view(1, -1, 1, 1).to(device), torch.tensor(std).view(1, -1, 1, 1).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        img = data['img']
        # Bug fix: use clone().detach().requires_grad_(True) instead of deprecated Variable
        inputs = img.clone().detach().requires_grad_(True)
        data['img'] = inputs
        output = func(**data)
        loss = criterion(output[head], data['gt_label'])
        loss.backward()

        sign_data_grad = inputs.grad.sign()
        # Bug fix: perturb directly in normalized space, then clip in pixel space
        img_pixel = inputs.detach() * std + mean
        img_pixel_adv = img_pixel + (eps / 255.) * sign_data_grad
        img_pixel_adv = torch.clamp(img_pixel_adv, 0, 1)
        inputs = (img_pixel_adv - mean) / std
        data['img'] = inputs.detach()

        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


r""" 
    PGD Adversarial Attack
"""
def pgd_nondist_forward_collect(func, data_loader, length, head, dataset='vera220', random_start=True, targeted=False):

    eps = 8
    alpha = 2
    steps = 10
    if dataset == 'vera220':
        mean=[0.4399, 0.4399, 0.4399]
        std=[0.114, 0.114, 0.114]
    elif dataset == 'tju600':
        mean=[0.382, 0.382, 0.382]
        std=[0.088, 0.088, 0.088]
    elif dataset == 'hkpu500':
        mean=[0.556, 0.556, 0.556]
        std=[0.047, 0.047, 0.047]
    elif dataset == 'casia200':
        mean=[0.471, 0.471, 0.471]
        std=[0.067, 0.067, 0.067]
    elif dataset == 'scut1100':
        mean=[0.288, 0.288, 0.288]
        std=[0.264, 0.264, 0.264]
    else: 
        raise ValueError("please chose a valid dataset")
    
    device = img.device
    mean, std = torch.tensor(mean).view(1, -1, 1, 1).to(device), torch.tensor(std).view(1, -1, 1, 1).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        img = data['img']
        inputs = img.clone().detach().requires_grad_(True)

        # denorm original image to pixel space once (used as clip center)
        img_pixel = img.detach() * std + mean

        if random_start:
            # Bug fix: rand_perturb is already in pixel space, do NOT multiply by std again
            rand_perturb = torch.rand_like(img_pixel) * 2 * (eps / 255.) - (eps / 255.)
            inputs_pixel = torch.clamp(img_pixel + rand_perturb, 0, 1)
            inputs = ((inputs_pixel - mean) / std).detach().requires_grad_(True)

        # PGD iter
        for _ in range(steps):
            data['img'] = inputs
            output = func(**data)
            if targeted:
                loss = -criterion(output[head], data['gt_label'])
            else:
                loss = criterion(output[head], data['gt_label'])
            loss.backward()

            sign_data_grad = inputs.grad.sign()
            # Bug fix: update in pixel space, clip, then re-normalize
            # img_pixel is computed once outside the loop (no drift)
            inputs_pixel = inputs.detach() * std + mean + (alpha / 255.) * sign_data_grad
            inputs_pixel = torch.max(torch.min(inputs_pixel, img_pixel + (eps / 255.)), img_pixel - (eps / 255.))
            inputs_pixel = torch.clamp(inputs_pixel, 0, 1)
            inputs = ((inputs_pixel - mean) / std).detach().requires_grad_(True)
        data['img'] = inputs.detach()

        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


r""" 
    Auto Adversarial Attack
    Reference: https://github.com/fra31/auto-attack
"""
def _l0_norm(x):
    return (x.abs().view(x.shape[0], -1) > 0).sum(dim=-1)


def _l1_norm(x, keepdim=False):
    out = x.abs().view(x.shape[0], -1).sum(dim=-1)
    return out.view(-1, 1, 1, 1) if keepdim else out


def _l2_norm(x, keepdim=False):
    out = (x ** 2).view(x.shape[0], -1).sum(dim=-1).sqrt()
    return out.view(-1, 1, 1, 1) if keepdim else out


def _check_zero_gradients(grad, logger=None):
    # lightweight compatibility with auto-attack; keep silent by default
    if grad is None:
        return
    with torch.no_grad():
        g = grad.detach()
        is_zero = (g.abs().view(g.shape[0], -1).sum(dim=-1) == 0)
        if is_zero.any() and logger is not None:
            logger.warning(f'found {int(is_zero.sum())} samples with zero gradients')


def _l1_projection(x2, y2, eps1):
    """Projection used by AutoAttack APGD for L1 norm.

    This is adapted from `autoattack/autopgd_base.py` (same math/steps).
    """
    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement() != 0:
        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)
        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter = 0
        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.0)
            counter2 = counter4.long()  # Bug fix: keep on same device as s/c (avoid CPU LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            if ind3.nelement() != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement() != 0:
                ub[ind32] = counter4[ind32]
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


def _apgd_check_oscillation(x, j, k, device, k3=0.75):
    t = torch.zeros(x.shape[1], device=device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()
    return (t <= k * k3 * torch.ones_like(t)).float()


def _apgd_normalize(delta, norm, ndims):
    if norm == 'Linf':
        t = delta.abs().view(delta.shape[0], -1).max(1)[0]
    elif norm == 'L2':
        t = (delta ** 2).view(delta.shape[0], -1).sum(-1).sqrt()
    elif norm == 'L1':
        try:
            t = delta.abs().view(delta.shape[0], -1).sum(dim=-1)
        except Exception:
            t = delta.abs().reshape([delta.shape[0], -1]).sum(dim=-1)
    else:
        raise ValueError('unknown norm')
    return delta / (t.view(-1, *([1] * ndims)) + 1e-12)


def _apgd_dlr_loss(logits, y):
    x_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(logits.shape[0], device=logits.device)
    return -(logits[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
        x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def apgd_attack_single_run(predict, x, y, eps, n_iter=100, norm='Linf', loss='ce', eot_iter=1,
                           rho=0.75, x_init=None, logger=None):
    """Core APGD single-run attack (evaluation only).

    Adapted from `autoattack/autopgd_base.py: APGDAttack.attack_single_run`.
    Inputs are pixel-space images in [0,1]. `predict(x)` returns logits.
    """
    assert norm in ['Linf', 'L2', 'L1']
    # assert loss in ['ce', 'dlr', 'ce-targeted-cfts']
    # TODO: support Cross-Entropy loss only for now
    assert loss in ['ce']

    device = x.device
    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)

    # checkpoint schedule
    n_iter_2 = max(int(0.22 * n_iter), 1)
    n_iter_min = max(int(0.06 * n_iter), 1)
    size_decr = max(int(0.03 * n_iter), 1)

    if len(x.shape) == ndims:  # no batch dim
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if norm == 'Linf':
        t = 2 * torch.rand(x.shape, device=device).detach() - 1
        x_adv = x + eps * torch.ones_like(x).detach() * _apgd_normalize(t, norm, ndims)
    elif norm == 'L2':
        t = torch.randn(x.shape, device=device).detach()
        x_adv = x + eps * torch.ones_like(x).detach() * _apgd_normalize(t, norm, ndims)
    else:  # L1
        t = torch.randn(x.shape, device=device).detach()
        delta = _l1_projection(x, t, eps)
        x_adv = x + t + delta

    if x_init is not None:
        x_adv = x_init.clone()

    x_adv = x_adv.clamp(0.0, 1.0)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)

    if loss == 'ce':
        criterion_indiv = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError("please chose a valid loss")
    # elif loss == 'ce-targeted-cfts':
    #     criterion_indiv = lambda xlog, ylab: -1.0 * F.cross_entropy(xlog, ylab, reduction='none')
    # else:  # dlr
    #     criterion_indiv = _apgd_dlr_loss


    # initial gradient
    x_adv.requires_grad_()
    grad = torch.zeros_like(x)
    for _ in range(eot_iter):
        with torch.enable_grad():
            logits = predict(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss_sum = loss_indiv.sum()
        grad += torch.autograd.grad(loss_sum, [x_adv])[0].detach()
    grad /= float(eot_iter)
    grad_best = grad.clone()

    if loss in ['dlr']:
        _check_zero_gradients(grad, logger=logger)

    acc = logits.detach().max(1)[1] == y
    loss_best = loss_indiv.detach().clone()

    alpha = 2.0 if norm in ['Linf', 'L2'] else 1.0
    step_size = alpha * eps * torch.ones([x.shape[0], *([1] * ndims)], device=device).detach()
    x_adv_old = x_adv.detach().clone()
    k = n_iter_2
    n_fts = math.prod(orig_dim)
    if norm == 'L1':
        k = max(int(.04 * n_iter), 1)
        if x_init is None:
            topk = .2 * torch.ones([x.shape[0]], device=device)
            sp_old = n_fts * torch.ones_like(topk)
        else:
            topk = _l0_norm(x_adv - x) / n_fts / 1.5
            sp_old = _l0_norm(x_adv - x)
        adasp_redstep = 1.5
        adasp_minstep = 10.0
    counter3 = 0

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    u = torch.arange(x.shape[0], device=device)

    for i in range(n_iter):
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            a = 0.75 if i > 0 else 1.0

            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - eps), x + eps), 0.0, 1.0)
            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * _apgd_normalize(grad, norm, ndims)
                x_adv_1 = torch.clamp(
                    x + _apgd_normalize(x_adv_1 - x, norm, ndims) * torch.min(
                        eps * torch.ones_like(x).detach(),
                        _l2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(
                    x + _apgd_normalize(x_adv_1 - x, norm, ndims) * torch.min(
                        eps * torch.ones_like(x).detach(),
                        _l2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
            else:  # L1
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (_l1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                delta_u = x_adv_1 - x
                delta_p = _l1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p

            x_adv = x_adv_1 + 0.0

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(eot_iter):
            with torch.enable_grad():
                logits = predict(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss_sum = loss_indiv.sum()
            grad += torch.autograd.grad(loss_sum, [x_adv])[0].detach()
        grad /= float(eot_iter)

        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        ind_pred = (pred == 0).nonzero().squeeze()
        if ind_pred.numel() != 0:
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.0

        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            if ind.numel() != 0:
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1
            if counter3 == k:
                if norm in ['Linf', 'L2']:
                    fl_oscillation = _apgd_check_oscillation(loss_steps, i, k, device=device, k3=rho)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - size_decr, n_iter_min)
                else:  # L1
                    sp_curr = _l0_norm(x_best - x)
                    fl_redtopk = (sp_curr / sp_old) < .95
                    topk = sp_curr / n_fts / 1.5
                    step_size[fl_redtopk] = alpha * eps
                    step_size[~fl_redtopk] /= adasp_redstep
                    step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                    sp_old = sp_curr.clone()
                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                counter3 = 0

    return x_best_adv.detach()


def apgd_perturb(predict, x, y, eps, n_iter=100, n_restarts=1, norm='Linf', loss='ce', eot_iter=1,
                 rho=0.75, seed=0, x_init=None, logger=None):
    """APGD multi-restart wrapper (evaluation only).

    Adapted from `autoattack/autopgd_base.py: APGDAttack.perturb` (non-targeted).
    Returns adversarial examples in [0,1].
    """
    device = x.device
    x = x.detach().clone().float().to(device)
    y = y.detach().clone().long().to(device)

    with torch.no_grad():
        y_pred = predict(x).max(1)[1]
    acc = y_pred == y
    adv = x.clone()

    torch.random.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(int(seed))

    for _ in range(int(n_restarts)):
        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() == 0:
            break

        x_to_fool = x[ind_to_fool].clone()
        y_to_fool = y[ind_to_fool].clone()
        adv_curr = apgd_attack_single_run(
            predict=predict, x=x_to_fool, y=y_to_fool, eps=eps,
            n_iter=n_iter, norm=norm, loss=loss, eot_iter=eot_iter,
            rho=rho, x_init=x_init, logger=logger,
        )

        with torch.no_grad():
            is_adv = predict(adv_curr).max(1)[1] != y_to_fool
            if is_adv.any():
                adv[ind_to_fool[is_adv]] = adv_curr[is_adv]
                acc[ind_to_fool[is_adv]] = 0

    return adv


def apgd_nondist_forward_collect(func, data_loader, length, head, dataset='vera220', eps=8,
                                steps=100, n_restarts=1, norm='Linf', loss='ce', seed=0, rho=0.75):
    """AutoPGD(APGD) adversarial forward & collect (non-distributed).

    `eps` follows existing FGSM/PGD convention (pixel-space, divided by 255).
    """
    if dataset == 'vera220':
        mean = [0.4399, 0.4399, 0.4399]
        std = [0.114, 0.114, 0.114]
    elif dataset == 'tju600':
        mean = [0.382, 0.382, 0.382]
        std = [0.088, 0.088, 0.088]
    elif dataset == 'hkpu500':
        mean = [0.556, 0.556, 0.556]
        std = [0.047, 0.047, 0.047]
    elif dataset == 'casia200':
        mean = [0.471, 0.471, 0.471]
        std = [0.067, 0.067, 0.067]
    elif dataset == 'scut1100':
        mean = [0.288, 0.288, 0.288]
        std = [0.264, 0.264, 0.264]
    else:
        raise ValueError("please chose a valid dataset")

    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    eps_f = float(eps) / 255.0

    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for _, data in enumerate(data_loader):
        img_norm = data['img']
        y = data['gt_label']
        device = img_norm.device
        mean = mean_t.to(device=device, dtype=img_norm.dtype)
        std = std_t.to(device=device, dtype=img_norm.dtype)

        # denorm to pixel space [0,1]
        x = (img_norm * std + mean).clamp(0.0, 1.0)

        def predict(x_pixel):
            x_in = (x_pixel - mean) / std
            data_adv = dict(data)
            data_adv['img'] = x_in
            out = func(**data_adv)
            return out[head]

        x_adv = apgd_perturb(predict=predict, x=x, y=y, eps=eps_f, n_iter=steps, n_restarts=n_restarts,
            norm=norm, loss=loss, eot_iter=1, rho=rho, seed=seed, x_init=None, logger=None,)
        data['img'] = ((x_adv - mean) / std).detach()

        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate([batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all
