# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ..registry import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .label_smooth_loss import LabelSmoothLoss
from .utils import convert_to_one_hot


@LOSSES.register_module()
class BootstrappingLoss(nn.Module):
    """Bootstrapping Loss.
    
    Ref: "Training Deep Neural Networks on Noisy Labels with Bootstrapping", 
    ICLR 2015.
    
    Args:
        bootstrap_beta (float): The weight of the original label. 
            The target becomes: beta * label + (1 - beta) * prediction.
        num_classes (int, optional): The number of classes. Defaults to None.
        mode (str): 'soft' or 'hard'. 'soft' uses the prediction distribution,
            'hard' uses the argmax of the prediction. Defaults to 'soft'.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 beta,
                 num_classes=None,
                 mode='soft',
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.beta = beta

        assert 0 <= beta <= 1, \
            f'BootstrappingLoss accepts beta over [0, 1], ' \
            f'but gets {beta}'

        accept_reduction = {'none', 'mean', 'sum'}
        assert reduction in accept_reduction, \
            f'BootstrappingLoss supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

        accept_mode = {'soft', 'hard'}
        assert mode in accept_mode, \
            f'BootstrappingLoss supports mode {accept_mode}, but gets {mode}.'
        self.mode = mode

        self.ce = CrossEntropyLoss(use_soft=True)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1), self.num_classes)
        return label.float()

    def bootstrap_label(self, one_hot_like_label, cls_score):
        """This function generates the enhanced label"""
        pred_softmax = F.softmax(cls_score, dim=1)

        if self.mode == 'soft':
            # y_target = beta * y_true + (1 - beta) * p
            target = self.beta * one_hot_like_label + \
                     (1 - self.beta) * pred_softmax
        else:
            # y_target = beta * y_true + (1 - beta) * one_hot(argmax(p))
            pred_hard = torch.zeros_like(pred_softmax).scatter_(
                1, pred_softmax.argmax(dim=1, keepdim=True), 1.0)
            target = self.bootstrap_beta * one_hot_like_label + \
                     (1 - self.bootstrap_beta) * pred_hard
        
        return target

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is None:
            self.num_classes = cls_score.shape[1]
        
        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'BootstrappingLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'
            
        with torch.no_grad():
            enhanced_label = self.bootstrap_label(one_hot_like_label, cls_score)

        return self.loss_weight * self.ce.forward(
            cls_score,
            enhanced_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)


@LOSSES.register_module()
class ConfidencePenaltyLoss(nn.Module):

    def __init__(self,
                 lambda_cp=0.1,
                 num_classes=None,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.lambda_cp = lambda_cp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ce = CrossEntropyLoss(use_soft=True)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1), self.num_classes)
        return label.float()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is None:
            assert cls_score.dim() == 2
            self.num_classes = cls_score.shape[1]
        else:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'

        log_probs = F.log_softmax(cls_score, dim=1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=1)
        loss_cp = self.lambda_cp * entropy
        
        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'
        
        return self.ce.forward(
            cls_score,
            one_hot_like_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs) - loss_cp


@LOSSES.register_module()
class TemperatureSmoothLoss(LabelSmoothLoss):
    def __init__(self,
                 temperature=0.5,
                 **kwargs):
        super().__init__()
        self.temp = temperature

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is None:
            assert cls_score.dim() == 2
            self.num_classes = cls_score.shape[1]
        else:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'

        smoothed_label = self.smooth_label(one_hot_like_label)

        # Temperature-based scaling
        cls_score = cls_score / self.temp

        return self.ce.forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)


@LOSSES.register_module()
class DirichletLabelSmoothLoss(nn.Module):
    """Dirichlet Label Smoothing Loss.
    
    Args:
        alpha (float): Parameter for Dirichlet distribution. 
            Typical range [0.1, 1.0]. Lower means more peaky.
        label_smooth_val (float): The total epsilon to be redistributed.
        num_classes (int, optional): The number of classes.
    """
    def __init__(self,
                 alpha=0.1,
                 label_smooth_val=0.1,
                 num_classes=None,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self._eps = label_smooth_val
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        self.ce = CrossEntropyLoss(use_soft=True)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1), self.num_classes)
        return label.float()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is None:
            assert cls_score.dim() == 2
            self.num_classes = cls_score.shape[1]
        else:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        
        batch_size = cls_score.shape[0]
        device = cls_score.device

        # 1. generate one-hot label
        one_hot_like_label = self.generate_one_hot_like_label(label=label)

        # 2. sample noise from Dirichlet distribution
        dist = torch.distributions.dirichlet.Dirichlet(
            torch.full((self.num_classes,), self.alpha, device=device))
        noise = dist.sample((batch_size,))  # [N, K]

        # 3. mix label: (1-eps) * y_true + eps * noise
        smoothed_label = (1 - self._eps) * one_hot_like_label + self._eps * noise

        return self.ce.forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)


@LOSSES.register_module()
class OnlineLabelSmoothLoss(nn.Module):
    """Online Label Smoothing Loss.
    
    Ref: "Online Label Smoothing", CVPR 2021.
    
    Args:
        alpha (float): The weight of the online soft label (1-alpha is hard label).
        num_classes (int): Necessary for buffer initialization.
    """
    def __init__(self,
                 num_classes,
                 alpha=0.1,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.loss_weight = loss_weight
        
        self.ce = CrossEntropyLoss(use_soft=True)

        # maintain a full-sized matrix of category probability distribution statistics [K, K]
        # each row represents the "expected output distribution" of the sample of the category in the model's eyes
        self.register_buffer('soft_label_matrix', 
                             torch.ones(num_classes, num_classes) / num_classes)
    
    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1), self.num_classes)
        return label.float()

    def _get_label_indices(self, label):
        # if the label is one-hot encoded (shape [N, K]), convert it to category indices (shape [N])
        if label.dim() == 2 and label.shape[1] == self.num_classes:
            # extract the category index from the one-hot encoded label (long-type)
            label_indices = torch.argmax(label, dim=1).long()
        # if the label is in index form (shape [N]), convert it to long-type
        else:
            label_indices = label.view(-1).long()
        return label_indices

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is None:
            assert cls_score.dim() == 2
            self.num_classes = cls_score.shape[1]
        else:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        label_indices = self._get_label_indices(label)

        batch_size = cls_score.shape[0]
        
        # 1. get the online soft label for the current batch
        with torch.no_grad():
            online_soft_label = self.soft_label_matrix[label_indices] # [N, K]
            
            # 2. construct the fused label: (1-alpha) * y_hard + alpha * y_online
            y_hard = one_hot_like_label
            smoothed_label = (1 - self.alpha) * y_hard + self.alpha * online_soft_label

            # 3. update the global soft label matrix (Momentum Update is optional, here we use the simple moving average idea)
            # in actual engineering, OLS is usually updated at the end of an Epoch or through EMA
            pred_prob = F.softmax(cls_score.detach(), dim=1)
            for i in range(batch_size):
                self.soft_label_matrix[label_indices[i]] = 0.9 * self.soft_label_matrix[label_indices[i]] + 0.1 * pred_prob[i]

        return self.ce.forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)