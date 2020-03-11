#!/usr/bin/env python3

import numpy as np
import torch

import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import model_builder
from utils.meters import TestMeter

logger = logging.get_logger(__name__)

def multi_view_test(test_loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx) in enumerate(test_loader):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()

        if cfg.TEST.IS_CAM_TEST == True:
            preds,feat,fc_w = model(inputs)
        else:
            preds = model(inputs)
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])

        test_meter.iter_toc()

        test_meter.update_stats(
            preds.detach().cpu(),
            labels.detach().cpu(),
            video_idx.detach().cpu(),
        )
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
    test_meter.finalize_metrics()
    test_meter.reset()

def test(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging()

    logger.info("Test with config:")
    logger.info(cfg)

    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE=="caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS >1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert(
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )

    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
    )

    multi_view_test(test_loader, model, test_meter, cfg)

    
