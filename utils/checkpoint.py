import os
import pickle
from collections import OrderedDict
import torch

import utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    checkpoint_dir = os.path.join(path_to_job,"checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_checkpoint_dir(path_to_job):
    return os.path.join(path_to_job, "checkpoints")

def get_path_to_checkpoint(path_to_job, epoch):
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job),name)

def get_last_checkpoint(path_to_job):
    d = get_checkpoint_dir(path_to_job)
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    name = sorted(names)[-1]
    return os.path.join(d, name)

def has_checkpoint(path_to_job):
    d = get_checkpoint_dir(path_to_job)

    files = os.listdir(d) if os.path.exists(d) else []

    return any("checkpoint" in f for f in files)

def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    return (cur_epoch + 1) % checkpoint_period == 0

def save_checkpoint(path_to_job, model, optimizer, epoch, cfg):
    os.makedirs(get_checkpoint_dir(path_to_job), exist_ok=True)
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint

def inflate_weight(state_dict_2d, state_dict_3d):
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            assert v2d.shape[-2:]== v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1,1,v3d.shape[2], 1,1)/v3d.shape[2]
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated

def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
):
    assert os.path.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)

    ms = model.module if data_parallel else model
    
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    if inflation:
        model_state_dict_3d = (
            model.module.state_dict() if data_parallel else model.state_dict()
        )
        inflated_model_dict = inflate_weight(
            checkpoint["model_state"],model_state_dict_3d
        )
        ms.load_state_dict(inflated_model_dict, strict=False)
    else:
        ms.load_state_dict(checkpoint["model_state"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    return epoch