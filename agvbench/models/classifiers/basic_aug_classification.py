import os
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from mmcv.runner import force_fp32, load_checkpoint
from agvbench.utils import print_log
from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..augments.basic import (cutout, gridmask, ricap, yoco, spnoise, randomblur, randnquant,
                              keepaugment, softaugment, smdwt_pca)
from ..utils import PlotTensor


@MODELS.register_module
class BasicAugClassification(BaseModel):
    """Basic Augmentation classification.

    Args:
        backbone (dict): Config dict for module of a backbone architecture.
        head (dict): Config dict for module of loss functions.
        backbone_k (dict, optional): Config dict for pre-trained backbone. Default: None.
        alpha (float or list): To sample Beta distribution in Augmentation methods. Build a
            list for various augmentation methods. Default: 1.
        aug_mode (str or list): Basice augmentation methods in input space. Similarly, build
            a list for various aug_mode, and randomly choose one aug_mode for each iter.
            Default: "augmentation".
        aug_args (dict): Dict of args (hyper-parameters) for various augmentation methods.
        aug_prob (list, optional): List of applying prob for given augmentation modes. Default: None.
        aug_repeat (bool or int, optional): How many time to repeat augmentation within a mini-batch.
            If aug_repeat > 1, augmentation with different alpha and shuffle idx. Default: False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for backbone_k or
    """

    def __init__(self,
                 backbone,
                 head=None,
                 backbone_k=None,
                 alpha=1.0,
                 aug_mode="cutout",
                 aug_args=dict(),
                 aug_prob=None,
                 aug_repeat=False,
                 pretrained=None,
                 pretrained_k=None,
                 save_name='AugedSamples',
                 debug_mode=True,
                 init_cfg=None,
                 **kwargs):
        super(BasicAugClassification, self).__init__(init_cfg, **kwargs)
        # networks
        assert isinstance(backbone, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        self.backbone_k = None
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            for param in self.backbone_k.parameters():  # stop grad k
                param.requires_grad = False

        # augmentation args
        self.aug_mode = aug_mode if isinstance(aug_mode, list) else [str(aug_mode)]
        self.masking_mode = {
            "cutout": cutout, "gridmask": gridmask, "spnoise": spnoise, "randomblur": randomblur,
            "randnquant": randnquant, "smdwt_pca": smdwt_pca
        }
        self.cutting_mode = {
            "ricap": ricap, "yoco": yoco
        }
        self.policy_mode = {
            "softaugment": softaugment, "keepaugment": keepaugment,
        }
        self.aug_args = dict(  # default settings
            cutout=dict(),
            gridmask=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
                         cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
            spnoise=dict(prob=0.1, noise_type='random'),
            randomblur=dict(),
            randnquant=dict(region_num=4, collapse_to_val='inside_random', spacing='random'),
            ricap=dict(choose_num=2,),
            yoco=dict(),
            softaugment=dict(t_crop=1.0, max_p_crop=1.0, pow_crop=2.0, bg_crop=1, sigma_crop=12,
                        iou=False, n_classes=220),
            smdwt_pca=dict(thresholds=(0.55, 0.65), wavelet=('bior1.3', 'bior4.4', 'bior6.8')),
            keepaugment=dict(threshold=0.5, mode='paste', randaugment_n=2, randaugment_m=9),
            vanilla=dict(),
        )
        _supported_mode = ["vanilla"] + list(self.masking_mode.keys()) + list(self.cutting_mode.keys()) + list(self.policy_mode.keys())
        for _mode in _supported_mode:
            self.aug_args[_mode].update(aug_args.get(_mode, dict()))  # update aug_args
        for _mode in self.aug_mode:
            assert _mode in _supported_mode, "The aug_mode={} is not supported!".format(_mode)
        self.alpha = alpha if isinstance(alpha, list) else [float(alpha)]
        assert len(self.alpha) == len(self.aug_mode) and len(self.aug_mode) < 6
        self.idx_list = [i for i in range(len(self.aug_mode))]
        self.aug_prob = aug_prob if isinstance(aug_prob, list) else None
        if self.aug_prob is not None:
            assert len(self.aug_prob) == len(self.alpha) and abs(sum(self.aug_prob) - 1e-10) <= 1, \
                "aug_prob={}, sum={}, alpha={}".format(self.aug_prob, sum(self.aug_prob), self.alpha)
        self.aug_repeat = int(aug_repeat) if int(aug_repeat) > 1 else 1
        if self.aug_repeat > 1:
            print_log("Warning: aug_repeat={} is more than once.".format(self.aug_repeat))
        if len(self.aug_mode) < self.aug_repeat:
            print_log("Warning: the number of aug_mode={} is less than aug_repeat={}.".format(
                self.aug_mode, self.aug_repeat))
        self.debug_mode = debug_mode
        self.save_name = str(save_name)
        self.save = False
        self.ploter = PlotTensor(apply_inv=True)
        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights. Default: None.
            pretrained_k (str, optional): Path to pre-trained weights for encoder_k.
                Default: None.
        """
        if self.init_cfg is not None:
            super(BasicAugClassification, self).init_weights()

        # init pre-trained params
        if pretrained_k is not None:
            print_log('load pre-training from: {}'.format(pretrained_k), logger='root')
            if self.backbone_k is not None:
                self.backbone_k.init_weights(pretrained=pretrained_k)
        # init trainable params
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
            load_checkpoint(self, pretrained, strict=False, logger=logging.getLogger())
            self.backbone.init_weights(pretrained=pretrained)
            self.head.init_weights()
        if self.backbone_k is not None and pretrained_k is None:
            for param_q, param_k in zip(self.backbone.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)

    @force_fp32(apply_to=('img',))
    def forward_aug(self, img, gt_label, remove_idx=-1):
        """computate mini-batch augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            remove_idx (int): Remove this idx this time.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # choose a augmentation method
        if self.aug_prob is None:
            candidate_list = self.idx_list.copy()
            if 0 <= remove_idx <= len(self.idx_list):
                candidate_list.remove(int(remove_idx))
            cur_idx = random.choices(candidate_list, k=1)[0]
        else:
            candidate_list = self.idx_list.copy()
            if 0 <= remove_idx <= len(self.idx_list):
                candidate_list.remove(int(remove_idx))
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            cur_idx = random_state.choice(candidate_list, p=self.aug_prob)
        cur_mode, cur_alpha = self.aug_mode[cur_idx], self.alpha[cur_idx]

        return_mask, mask = False, None  # return sample mask in [N, 1, H, W]
        
        # applying masking sample augmentation methods
        if cur_mode in ["cutout", "gridmask", "spnoise", "randomblur", "randnquant", "smdwt_pca"]:
            if cur_mode in ["cutout", "gridmask", "randomblur"]:
                img = self.masking_mode[cur_mode](img, cur_alpha, dist_mode=False, 
                                                return_mask=return_mask, **self.aug_args[cur_mode])
            elif cur_mode in ["spnoise", "randnquant", "smdwt_pca"]:
                img = self.masking_mode[cur_mode](img, dist_mode=False, 
                                                return_mask=return_mask, **self.aug_args[cur_mode])
            if return_mask:
                img, mask = img  # (img, mask): get mask
        elif cur_mode in ["ricap", "yoco"]:
            if cur_mode == 'yoco':
                img = self.cutting_mode[cur_mode](img, cur_alpha, dist_mode=False, 
                                                  return_mask=return_mask, **self.aug_args[cur_mode])
            elif cur_mode == 'ricap':
                img, gt_label = self.cutting_mode[cur_mode](img, gt_label, cur_alpha, dist_mode=False, 
                                                            return_mask=return_mask, **self.aug_args[cur_mode])
        elif cur_mode in ["softaugment", "keepaugment"]:
            if cur_mode == 'softaugment':
                img = self.policy_mode[cur_mode](img, **self.aug_args[cur_mode])
            elif cur_mode == 'keepaugment':
                pred_raw = self.backbone(img)[0].clone().detach()
                img = self.policy_mode[cur_mode](img, gt_label, pred_raw, **self.aug_args[cur_mode])
        else:
            assert cur_mode == "vanilla"
        x = self.backbone(img)

        # augmentation loss
        pred_aug = self.head(x)
        losses = self.head.loss(pred_aug, gt_label)
        losses['loss'] /= self.aug_repeat

        # save augmented img
        if self.save:
            self.plot_aug(img_aug=img, aug_mode=cur_mode)

        if self.debug_mode:
            if torch.any(torch.isnan(losses['loss'])) or torch.any(torch.isinf(losses['loss'])):
                raise ValueError("Inf or nan value: use FP32 instead.")

        return losses, cur_idx

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if isinstance(img, list):
            img = img[0]

        # repeat aug within a mini-batch
        losses = dict()
        remove_idx = -1
        for i in range(self.aug_repeat):
            if i == 0:
                losses, cur_idx = self.forward_aug(img.clone(), gt_label, remove_idx=remove_idx)
            else:
                _loss, cur_idx = self.forward_aug(img.clone(), gt_label, remove_idx=remove_idx)
                losses["loss"] += _loss["loss"]
            # remove 'vanilla' if chosen
            if self.aug_mode[cur_idx] == "vanilla":
                remove_idx = cur_idx

        return losses


    def simple_test(self, img):
        """Test without augmentation."""
        x = self.backbone(img)[-1:]
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def augment_test(self, img):
        """Test function with test time augmentation."""
        x = [self.backbone(_img)[-1] for _img in img]
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test(self, img, **kwargs):
        """
        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).
        """
        if isinstance(img, list):
            return self.augment_test(img)
        else:
            return self.simple_test(img)

    @force_fp32(apply_to=('img_aug',))
    def plot_aug(self, img_aug, aug_mode=""):
        """ visualize augmented results """
        img = torch.cat((img_aug[:4], img_aug[4:8], img_aug[8:12]), dim=0)
        title_name = "{}".format(aug_mode) \
            # if isinstance(lam, float) else aug_mode
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=self.save_name)