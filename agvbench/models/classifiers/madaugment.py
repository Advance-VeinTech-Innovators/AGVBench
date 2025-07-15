import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
import PIL.Image
import random
import logging
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from torchvision.utils import save_image
from agvbench.utils import print_log
from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..utils import PlotTensor

@MODELS.register_module
class MAdAugmentation(BaseModel):
    r""" MAdAugment.

    "Mixed Automatic Adversarial Augmentation Network for Finger-Vein Recognition. 
    (https://xplorestaging.ieee.org/document/10530126)". In IEEE TIM, 2024.
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet (main).
        backbone_k (dict): Config dict for module of momentum backbone ConvNet. Default: None.
        aug_module (dict): Config dict for the augmentation sub-network. Default: None.
        head_aug (dict): Config dict for module of augmented classification loss (aug_module).
        head_one (dict): Config dict for module of onehot classification loss (backbone).
        head_weights (dict): Dict of the used cls heads names and loss weights,
            which determines the cls or augmented head in used.
            Default: dict(head_one_q=1, head_aug_k=1)
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        head_ensemble (bool): Whether to ensemble results of all heads. Default to False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for en_k. Default: None.
        save_by_sample (bool): Whether to save mixup samples separately.
        debug_mode (bool): Whether to save some intermediate products.
    """
    def __init__(self,
                 backbone,
                 backbone_k=None,
                 aug_module=None,
                 beta=0.5,
                 head_one=None,
                 head_aug=None,
                 head_weights=dict(decent_weight=[], accent_weight=[],
                                   head_one_q=1, head_aug_k=1),
                 head_ensemble=False,
                 save=True,
                 save_name='MixedSamples',
                 debug=False,
                 pretrained=None,
                 pretrained_k=None,
                 init_cfg=None,
                 **kwargs):
        super(MAdAugmentation, self).__init__(init_cfg, **kwargs)
        # basic params
        self.head_ensemble = bool(head_ensemble)
        self.save = bool(save)
        self.save_name = str(save_name)
        self.ploter = PlotTensor(apply_inv=True)
        self.debug = bool(debug)
        self.beta = float(beta)
        self.iter = 0
        self.cos_simi = torch.nn.CosineSimilarity(dim=1)

        # network
        assert backbone_k is None or isinstance(backbone_k, dict)
        assert head_one is None or isinstance(head_one, dict)
        assert head_aug is None or isinstance(head_aug, dict)
        # augmentation module
        self.aug_module = builder.build_head(aug_module)
        # backbone
        self.backbone_q = builder.build_backbone(backbone)
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            assert pretrained_k is not None
        else:
            self.backbone_k = builder.build_backbone(backbone)
        self.backbone = self.backbone_k  # for feature extract
        # aug cls head
        assert "head_aug_k" in head_weights.keys()
        self.head_aug_k = builder.build_head(head_aug)
        # onehot cls head
        if "head_one_q" in head_weights.keys():
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        # for feature extract
        self.head = self.head_one_q
        self.weight_aug_k = head_weights.get("head_aug_k", 1.)
        self.weight_one_q = head_weights.get("head_one_q", 1.)
        self.head_weights = head_weights
        self.head_weights['decent_weight'] = head_weights.get("decent_weight", list())
        self.head_weights['accent_weight'] = head_weights.get("accent_weight", list())
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine

        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):

        if self.aug_module is not None:
            self.aug_module.init_weights(init_linear='normal')
        # init pretrained backbone_k and mixblock
        if pretrained_k is not None:
            print_log('load pretrained classifier k from: {}'.format(pretrained_k), logger='root')
            # load full ckpt to backbone and fc
            logger = logging.getLogger()
            load_checkpoint(self, pretrained_k, strict=False, logger=logger)

        # init backbone, based on params in q
        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.backbone_q.init_weights(pretrained=pretrained)

        # copy backbone param from q to k
        if pretrained_k is None:
            for param_q, param_k in zip(self.backbone_q.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # stop grad k

        # init head
        if self.head_one_q is not None:
            self.head_one_q.init_weights()
        if (self.head_one_q is not None and self.head_aug_k is not None) and \
                (pretrained_k is None):
            for param_one_q, param_aug_k in zip(self.head_one_q.parameters(),
                                                self.head_aug_k.parameters()):
                param_aug_k.data.copy_(param_one_q.data)
                param_aug_k.requires_grad = False  # stop grad k

    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        if self.cos_annealing < 0 or self.cos_annealing > 1:
            return
        # cos annealing decent, from 1 to 0
        if len(self.head_weights["decent_weight"]) > 0:
            for attr in self.head_weights["decent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * self.cos_annealing)
        # cos annealing accent, from 0 to 1
        if len(self.head_weights["accent_weight"]) > 0:
            for attr in self.head_weights["accent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * (1 - self.cos_annealing))

    @torch.no_grad()
    def weights_update(self):
        """Weights update of the k form q by hook, including the backbone and heads """
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_q.data

        if self.head_one_q is not None and self.head_aug_k is not None:
            for param_one_q, param_aug_k in zip(self.head_one_q.parameters(),
                                                self.head_aug_k.parameters()):
                param_aug_k.data = param_one_q.data

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if isinstance(img, list):
            img = img[0]
        self._update_loss_weights()
        self.weights_update()

        # Image AutoAugmentation
        if self.aug_module is not None:   # Need Write
            img_ = self.aug_module(img, gt_label, self.backbone_k, self.head_one_q)
        else:
            raise ValueError("MAdAugmentation need sub-module for augmentation.")

        # k (mb): the sub-module training
        loss_aug_k, loss_cos_k = self.forward_k(img, img_, gt_label)
        # q (bb): the encoder training
        loss_one_q, loss_aug_q = self.forward_q(img, img_.clone().detach(), gt_label)
        # save img mb
        if self.save:
            self.plot_aug(img_, img, "madaug")

        #  loss summary
        losses = {
            'loss': loss_aug_q['loss'] * self.weight_one_q,
            'acc_aug_q': loss_aug_q['acc'],
        }
        # onehot loss
        if loss_one_q is not None and self.weight_one_q > 0:
            losses['loss'] += loss_one_q['loss'] * self.weight_one_q
            losses['acc_one_q'] = loss_one_q['acc']
        # adversial training
        if loss_aug_k['loss'] is not None:
                loss_aug_k['loss'] = (-1.0) * loss_aug_k['loss'] + self.beta * loss_cos_k['loss']
                self.iter = 0
        # augmentation loss
        if loss_aug_k['loss'] is not None and self.weight_aug_k > 0:
            losses["loss"] += loss_aug_k['loss'] * self.weight_aug_k
            losses['acc_aug_k'] = loss_aug_k['acc']

        return losses


    @force_fp32(apply_to=('im_q', 'im_k'))
    def plot_aug(self, im_q, im_k, name="k"):
        # plot mixup results
        img = torch.cat((im_q[:4], im_k[:4]), dim=0)
        title_name = '{}'.format(name)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(img, nrow=4, title_name=title_name, save_name=self.save_name)

    @auto_fp16(apply_to=('x', 'aug_x'))
    def forward_q(self, x, aug_x, y):
        loss_one_q = None
        if self.head_one_q is not None and self.weight_one_q > 0:
            out_one_q = self.backbone_q(x)[-1]
            pred_one_q = self.head_one_q([out_one_q])
            # loss
            loss_one_q = self.head_one_q.loss(pred_one_q, y)
            if torch.isnan(loss_one_q['loss']):
                print_log("Warming NAN in loss_one_q. Please use FP32!", logger='root')
                loss_one_q = None
        # aug mixing samples -> q
        loss_aug_q = None
        if self.head_one_q is not None and self.weight_one_q > 0:
            out_aug_q = self.backbone_q(aug_x)[-1]
            pred_aug_q = self.head_one_q([out_aug_q])
            # loss
            loss_aug_q = self.head_one_q.loss(pred_aug_q, y)
            if torch.isnan(loss_aug_q['loss']):
                print_log("Warming NAN in loss_aug_q. Please use FP32!", logger='root')
                loss_aug_q = None

        return loss_one_q, loss_aug_q

    @auto_fp16(apply_to=('x', 'x_'))
    def forward_k(self, x, x_, y):

        loss_aug_k = None
        if self.weight_aug_k > 0:
            out_aug_k = self.backbone_k(x_)
            pred_aug_k = self.head_aug_k([out_aug_k[-1]])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_aug_k[0] = pred_aug_k[0].type(torch.float32)
            loss_aug_k = self.head_aug_k.loss(pred_aug_k, y)
            if torch.isnan(loss_aug_k['loss']):
                print_log("Warming NAN in loss_aug_k. Please use FP32!", logger='root')
                loss_aug_k["loss"] = None

        loss_cos_k = None
        out_one_k = self.backbone_k(x_)
        loss_cos_k = (1 - self.cos_simi(out_one_k[-1], out_aug_k[-1]).mean())

        return loss_aug_k, loss_cos_k

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        keys = list()
        pred = list()
        # backbone
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
        if self.weight_aug_k > 0:
            pred.append(self.head_aug_k([last_k]))
            keys.append('acc_aug_k')

        # head q
        pred.append(self.head_one_q([last_q]))
        keys.append('acc_aug_q')
        # head ensemble
        if self.head_ensemble:
            pred.append([torch.stack(
                [pred[i][0] ** 2 for i in range(len(pred))]).mean(dim=0)])
            keys.append('acc_avg')

        out_tensors = [p[0].cpu() for p in pred]  # NxC

        return dict(zip(keys, out_tensors))


    def forward_test(self, img, **kwargs):
        """Forward computation during testing.

        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).

        Returns:
            dict[key, Tensor]: A dictionary of head names (key) and predictions.
        """
        return self.simple_test(img)

    def forward_inference(self, img, **kwargs):
        """Forward output for inference.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.

        Returns:
            tuple[Tensor]: final model outputs.
        """
        x = self.backbone(img)[-1]
        preds = self.head([x], post_process=True)
        return preds[0]