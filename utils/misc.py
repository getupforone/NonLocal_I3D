import logging
import math
import numpy as np
import os
from datetime import datetime
import torch
import utils.logging as logging

logger = logging.get_logger(__name__)

def check_nan_losses(loss):
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def gpu_mem_usage():
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)

def log_model_info(model):
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info("nvdia-smi")
    os.system("nvidia-smi")

def is_eval_epoch(cfg, cur_epoch):
    return (
        cur_epoch + 1 
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH