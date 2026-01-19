import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import BaseModule

from ..utils import (accuracy, accuracy_mixup, accuracy_semantic_softmax,
                     trunc_normal_init)
from ..registry import HEADS
from ..builder import build_loss
from .cls_head import BaseClsHead
from .cls_mixup_head import ClsMixupHead


@HEADS.register_module
class ClsFVRASHead(BaseClsHead):
    def __init__(self, **kwargs):
        super(ClsFVRASHead, self).__init__(**kwargs)

        # build a classification head
        assert self.hidden_dim is None
        self.sf = nn.Softmax(dim=-1)
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels * 2 * 2, self.in_channels)
            self.fc2 = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsFVRASHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 2).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 2).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.fc(x)
        x = self.fc2(x)
        if post_process:
            x = self.post_process(x)
        return x


@HEADS.register_module
class ClsPVHead(BaseClsHead):
    def __init__(self, **kwargs):
        super(ClsPVHead, self).__init__(**kwargs)

        # build a classification head
        assert self.hidden_dim is None
        self.sf = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels * 2 * 2, self.in_channels)
            self.fc2 = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsPVHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 2).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 2).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        if post_process:
            x = self.post_process(x)
        return x


@HEADS.register_module
class ClsAMPVHead(BaseClsHead):
    def __init__(self, **kwargs):
        super(ClsAMPVHead, self).__init__(**kwargs)

        # build a classification head
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsAMPVHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.drop(x)
        x = self.fc(x)
        return x



@HEADS.register_module
class ClsFVRASMixupHead(ClsMixupHead):
    def __init__(self, **kwargs):
        super(ClsFVRASMixupHead, self).__init__(**kwargs)

        # build a classification head
        self.sf = nn.Softmax(dim=-1)
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels * 2 * 2, self.in_channels)
            self.fc2 = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsFVRASHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 2).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 2).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.fc(x)
        x = self.fc2(x)
        if post_process:
            x = self.post_process(x)
        return x


@HEADS.register_module
class ClsPVMixupHead(ClsMixupHead):
    def __init__(self, **kwargs):
        super(ClsPVMixupHead, self).__init__(**kwargs)

        # build a classification head
        self.sf = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        if self.num_classes is not None:
            self.fc = nn.Linear(self.in_channels * 2 * 2, self.in_channels)
            self.fc2 = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsPVHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 2).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 2).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        if post_process:
            x = self.post_process(x)
        return x


@HEADS.register_module
class ClsAMPVMixupHead(ClsMixupHead):
    def __init__(self, **kwargs):
        super(ClsAMPVMixupHead, self).__init__(**kwargs)

        # build a classification head
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        if self.fc is None:
            return
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(ClsAMPVHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        if self.finetune:  # finetune for ViTs
            std = 2e-5
            init_linear = 'trunc_normal'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward_head(self, x, post_process=False):
        """" forward cls head with x in a shape of (X, \*) """
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            else:
                assert x.dim() in [2, 3, 4], \
                    "Tensor must has 2, 3 or 4 dims, got: {}".format(x.dim())
        x = self.drop(x)
        x = self.fc(x)
        return x