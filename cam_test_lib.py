import torch
import cv2
import glob
import matplotlib.pyplot as plt
import os

import numpy as np


import torch.nn as nn
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import model_builder

logger = logging.get_logger(__name__)
TARGET_DIR = "/root/workspace/NonLocal_I3D"
import multiprocessing as mp
def upsample(in_img, in_scale_factor= (2,32,32)):
    upsample = nn.Upsample( scale_factor = in_scale_factor, mode = "bilinear")
    
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
        # preds : (8, 2, 1, 1, 1)
        # feat : (8, 2048, 4, 7, 7)
        # fc_w : (2, 2048)
        preds,feat,fc_w = model(inputs)
        print("cam_test :preds dim  = {}".format(preds.shape))
        print("cam_test :feat dim  = {}".format(feat.shape))
        print("cam_test :fc_w dim  = {}".format(fc_w.shape))
        preds = torch.squeeze(preds)
        preds_idx = torch.argmax(preds, dim=1)
        # fc_w_wide : (8, 2048 )
        fc_w_wide = fc_w[preds_idx] 
        # feat : (8, 2048, 4, 7, 7)
        # feat_per : (8, 4, 7, 7, 2048)
        feat_per = feat.permute((0,2,3,4,1))
        # ouput : ( 4, 7, 7, 2048) * (2048) = (4,7,7)
        output = torch.matmul(feat_per[0],fc_w_wide[0])
        
        upsample2=nn.Upsample(scale_factor = (2,1,1), mode = 'trilinear')
        output_unsqz =torch.unsqueeze(output,0) 
        print("cam_test :output_unsqz dim  = {}".format(output_unsqz.shape))
        output_unsqz =torch.unsqueeze(output_unsqz,0)
        print("cam_test :output_unsqz dim  = {}".format(output_unsqz.shape))
#         print(output_unsqz.shape)
        up_img_temp = upsample2(output_unsqz)
        # up_img_temp : (8,7,7)
        #up_img_temp = upsample2(output)
        print("cam_test: up_im_temp dim = {}".format(up_img_temp.shape))

        upsample=nn.Upsample(scale_factor = (32,32), mode = 'bilinear')
        up_img_temp_sqz = torch.squeeze(up_img_temp,0)
        print("cam_test: up_img_temp_sqz dim = {}".format(up_img_temp_sqz.shape))
        up_img = upsample(up_img_temp_sqz)
        print("cam_test: up_img dim = {}".format(up_img.shape))
        up_img = torch.squeeze(up_img,0)
        print("cam_test: up_img dim = {}".format(up_img.shape))
        #return up_img
        
def cam_test(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging()
    #cfg.BATCH_SIZE = 1
    logger.info("run with config:")
    logger.info(cfg)

    model = model_builder.build_model(cfg)
    misc.log_model_info(model)
    if cu.has_checkpoint(TARGET_DIR):
        last_checkpoint = cu.get_last_checkpoint(TARGET_DIR)
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
    cam_img = cam_view_test(test_loader, model,  cfg)