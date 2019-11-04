# https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import smoothing


def _is_long(x):
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(logits, target, weight=None, ignore_index=-100, reduction='mean', smooth_eps=0.):
    """cross entropy loss with support for target distributions"""
    with torch.no_grad():
        if smooth_eps > 0:
            target = smoothing(logits, target, smooth_eps)
    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target):
        return F.cross_entropy(logits, target, weight, ignore_index=ignore_index, reduction=reduction)

    masked_indices = None
    num_classes = logits.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    # log-softmax of logits
    lsm = F.log_softmax(logits, dim=-1)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets and built-in label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=0.):
        super(CrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return cross_entropy(input, target, self.weight, self.ignore_index, self.reduction, self.smooth_eps)
