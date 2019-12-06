#!/usr/bin/env python3

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
#from datasets.kinetics import Kinetics
from datasets.davis import Davis

#_DATASET_CATALOG = {"kinetics": Kinetics, "davis": Davis}
_DATASET_CATALOG = {"davis": Davis}

def construct_loader(cfg, split):
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), "Dataset '{}' is not supported".format(dataset_name)

    dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader

def shuffle_dataset(loader, cur_epoch):
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)
