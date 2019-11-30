#!/usr/bin/env python3

import torch.nn as nn

_LOSSES = {"cross_entropy": nn.CrossEntropyLoss}

def get_loss_func(loss_name):
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
    