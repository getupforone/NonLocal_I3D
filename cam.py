#!/usr/bin/env python3

import numpy as np
import torch

import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import model_builder

logger = logging.get_logger(__name__)
TARGET_DIR = "./results/kstartv_i3d_nln_16_8_2_224"


def cam_view_test(test_loader, model, cfg):
    model.eval()

    for cur_iter, (inputs, labels, video_idx) in enumerate(test_loader):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()

        preds,feat,fc_w = model(inputs)

def cam(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging()

    logger.info("run with config:")
    logger.info(cfg)

    model = model_builder.build_model(cfg)
    misc.log_model_info(model)
    if cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert(
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    cam_view_test(test_loader, model,  cfg)