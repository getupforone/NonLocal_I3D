#!/usr/bin/env python3

import torch
from models.video_model_builder import ResNetModel

_MODEL_TYPES = {
    "i3d": ResNetModel,
    "slowonly": ResNetModel,
}

def build_model(cfg):
    assert(
        cfg.MODEL.ARCH in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.ARCH)

    assert(
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    model = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )

    return model
