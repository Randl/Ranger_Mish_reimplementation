import os
import shutil

import torch


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def is_bn(module):
    return isinstance(module, torch.nn.BatchNorm1d) or \
           isinstance(module, torch.nn.BatchNorm2d) or \
           isinstance(module, torch.nn.BatchNorm3d) or \
           isinstance(module, torch.nn.BatchNorm3d)


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint{}.pth.tar', local_rank=0, test=False):
    save_path = os.path.join(filepath, filename.format(local_rank))
    best_path = os.path.join(filepath, 'model_best{}.pth.tar'.format(local_rank))
    torch.save(state, save_path)
    if test:
        torch.load(save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def smoothing(out, y, smooth_eps):
    num_classes = out.shape[1]
    if smooth_eps == 0:
        return y
    my = onehot(y, num_classes).to(out)
    true_class, false_class = 1. - smooth_eps * num_classes / (num_classes - 1), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    return my
