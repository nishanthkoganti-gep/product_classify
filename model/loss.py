import torch.nn.functional as F


def nll_loss(output, target, weights):
    return F.nll_loss(output, target, weight=weights)


def cross_entropy(output, target, weights):
    return F.cross_entropy(output, target, weight=weights)
