#!/usr/bin/env python3
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import models.losses as losses
import models.optimizer as optim
import utils.checkpoint as cu
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from datasets import loader
from models import model_builder
from utils.meters import TrainMeter, ValMeter

import utils.distributed as du

from torch.utils.tensorboard import SummaryWriter

logger = logging.get_logger(__name__)
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/i3d_nln_8x1')

def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    # print("=======================================\n")
    # print("train_loader size = {}".format(data_size))
    # print("=======================================\n")
    running_loss = 0.0
    running_top1_err = 0.0
    for cur_iter, (inputs, labels, _) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size , cfg)
        optim.set_lr(optimizer, lr)

        # Perform the forward pass.
        if cfg.TEST.IS_CAM_TEST == True:
            preds,feat,fc_w = model(inputs)
        else:
            preds = model(inputs)

        # Explicitly declare reduction to mean
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        
        #print("shape of preds = {}".format(preds.shape))
        #print("shape of labels = {}".format(labels.shape))
        # Compute the loss
        loss = loss_fun(preds, labels)
        #print("shape of loss = {}".format(loss.shape))
        # Check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()
        # print("=======================================\n")
        # print("TRAIN: inputs shape = {}".format(inputs.shape))
        # print("TRAIN: preds shape = {}".format(preds.shape))
        # print("TRAIN: labels shape = {}".format(labels.shape))
        # print("=======================================\n")
        # Compute the erros.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()

        #running_loss += loss.item()
        running_loss += loss
        running_top1_err += top1_err
        if (cur_iter % 10 == 9) and (du.is_master_proc()):
            writer.add_scalar('train/loss', running_loss / 10, cur_epoch * data_size + cur_iter  )
            writer.add_scalar('train/top1_err', running_top1_err / 10, cur_epoch * data_size + cur_iter  )
            running_loss = 0.0
            running_top1_err = 0.0


        train_meter.iter_toc()

        train_meter.update_stats(
            top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _) in enumerate(val_loader):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        labels = labels.cuda()

        # Compute the predictions.
        if cfg.TEST.IS_CAM_TEST == True:
            preds,feat,fc_w = model(inputs)
        else:
            preds = model(inputs)
        # Compute the errors.
        # print("=======================================\n")
        # print("EVAL: inputs shape = {}".format(inputs.shape))
        # print("EVAL: preds shape = {}".format(preds.shape))
        # print("EVAL: labels shape = {}".format(labels.shape))
        # print("=======================================\n")
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))


        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the erros from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        val_meter.iter_toc()

        val_meter.update_stats(
            top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

def calculate_and_update_precise_bn(loader, model, num_iters=200):
    def _gen_loader():
        for inputs, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)
        # dummy_input = torch.Tensor(1, 3, 3, 224, 224)
        # writer.add_graph(model, (dummy_input, ))



    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg,"val")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        loader.shuffle_dataset(train_loader, cur_epoch)
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)





