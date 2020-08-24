#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import model_builder
from matplotlib import pyplot as plt
import cv2
import os

logger = logging.get_logger(__name__)
TARGET_DIR = "."
RESULT_IMG_DIR = "./img"
RESULT_MULTI_IMG_DIR = "./img"
import multiprocessing as mp
def upsample(in_img, in_scale_factor= (2,32,32)):
    upsample = nn.Upsample( scale_factor = in_scale_factor, mode = "bilinear")

def cam_view_test(test_loader, model, cfg, path_to_seq_imgs):
    model.eval()
    
    for cur_iter, (inputs, labels, video_idx) in enumerate(test_loader):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        print("cur_iter :{}".format(cur_iter))
        print("video_idx :{}".format(video_idx.item()))
        # preds : (8, 2, 1, 1, 1)
        # feat : (8, 2048, 4, 7, 7)
        # fc_w : (2, 2048)
        print("input shape = {}".format(inputs.shape))
        preds,feat,fc_w = model(inputs)
        print("preds shape = {}".format(preds.shape))
        
        print("feat shape = {}".format(feat.shape))
        print("fc_w shape = {}".format(fc_w.shape))
        preds = torch.squeeze(preds)
        #preds_idx = torch.argmax(preds, dim=1)
        preds_idx = torch.argmax(preds)
        print("preds_idx shape = {}".format(preds_idx.shape))
        print("preds_idx  = {}".format(preds_idx))
        # fc_w_wide : (8, 2048 )
        fc_w_wide = fc_w[preds_idx] 
        print("fc_w_wide shape = {}".format(fc_w_wide.shape))
        # feat : (8, 2048, 4, 7, 7)
        # feat_per : (8, 4, 7, 7, 2048)
        feat_per = feat.permute((0,2,3,4,1))
        feat_per = torch.squeeze(feat_per)
        # ouput : ( 4, 7, 7, 2048) * (2048) = (4,7,7)
        output = torch.matmul(feat_per,fc_w_wide)
        print("output shape = {}".format(output.shape))

        output_unsqz =torch.unsqueeze(output,0) 
        output_unsqz =torch.unsqueeze(output_unsqz,0)
        print(output_unsqz.shape)

        if output.shape[0] != cfg.DATA.NUM_FRAMES :
            scale_num = cfg.DATA.NUM_FRAMES//output.shape[0]
            upsample2=nn.Upsample(scale_factor = (scale_num,1,1), mode = 'trilinear')
            up_img_temp = upsample2(output_unsqz)
            # up_img_temp : (8,7,7)
            #up_img_temp = upsample2(output)
            print(up_img_temp.shape)
        else:
            up_img_temp = output_unsqz

        upsample=nn.Upsample(scale_factor = (32,32), mode = 'bilinear')
        up_img_temp_sqz = torch.squeeze(up_img_temp,0)
        print(up_img_temp_sqz.shape)
        up_img = upsample(up_img_temp_sqz)
        print(up_img.shape)
        up_img = torch.squeeze(up_img,0)
        print("up_img shape = {}".format(up_img.shape))
        print(up_img)
       
       
        up_img_num = up_img.shape[0]
        v_idx = video_idx.item()
        #p_idx = (int)(v_idx/(cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)) 
        p_idx = v_idx # we did not spatial clop for the test session
        print("".format())
        print("v_idx = {}/ p_idx = {}/ path_to_seq_imgs size = {}/size of DataLoader = {}".format(v_idx,p_idx, len(path_to_seq_imgs),len(test_loader)))
        shot_num = path_to_seq_imgs[p_idx]
        img_dir = os.path.join(RESULT_IMG_DIR,shot_num)
        if os.path.exists(img_dir):
            print("dir {} exists".format(img_dir))
        else:
            os.mkdir(img_dir)

        multi_img_dir = os.path.join(RESULT_MULTI_IMG_DIR,shot_num)
        if os.path.exists(multi_img_dir):
            print("dir {} exists".format(multi_img_dir))
        else:
            os.mkdir(multi_img_dir)
        print("shape of inputs = {}".format(inputs.shape))
        inputs_sq = torch.squeeze(inputs,0)
        inputs_per = inputs_sq.permute((1,2,3,0))
        input_img = ((inputs_per.cpu()).detach()).numpy()
        result_np_img = ((up_img.cpu()).detach()).numpy()
        labels  = ((labels.cpu()).detach()).numpy()
       
        
        print("result_np_img type is {} dtype is {}".format(type(result_np_img), result_np_img.dtype))
        fig, ax = plt.subplots()
        #fig = plt.figure()
        if labels[0] == 1:
            img_label = 'True'
        else:
            img_label = 'False'
        fig_title = "kstartv_i3d_nln_16_8_2_224_{}_{}".format(img_label,shot_num)
        plt.rcParams['figure.figsize'] = [16, 16]
        plt.title(fig_title)
        plt.axis('off')
        rows=2
        cols=4
        xlabels = ["xlabel", "(a)","(b)","(c)","(d)","(e)","(f)","(h)","(i)"]
        for img_idx in range(up_img_num):
            img_name = "{}_{}_{}_{}.jpg".format(shot_num,v_idx,cur_iter,img_idx)
            norm_img_name = "{}_{}_{}_{}_norm.jpg".format(shot_num,v_idx,cur_iter,img_idx)
            
            img_path = os.path.join(img_dir,img_name)
            norm_img_path = os.path.join(img_dir,norm_img_name)

            print("img_path : {}".format(img_path))
            #im_gray = (result_np_img[img_idx]).astype('uint8')
            im_gray = (result_np_img[img_idx])
            norm_img = (cv2.normalize(im_gray,im_gray,0,255,norm_type=cv2.NORM_MINMAX)).astype('uint8')
            im_color = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            #im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
            
            
            cv2.imwrite(img_path,im_color)
            cv2.imwrite(norm_img_path,norm_img)
            
            xlab_idx = img_idx % 8 +1
            ax = fig.add_subplot(rows, cols, xlab_idx)
            ax.imshow(input_img[img_idx], alpha=0.7)
            ax.imshow(result_np_img[img_idx], cmap='jet', alpha=0.3)
            ax.set_xlabel(xlabels[xlab_idx])
            ax.set_xticks([]), ax.set_yticks([])    

            # ax.set_title(img_name) 
        #plt.show()
        multi_img_title = "{}.jpg".format(fig_title)
        multi_img_path = os.path.join(multi_img_dir,multi_img_title)
        print("multi_img_path = {}".format(multi_img_path))
        plt.savefig(multi_img_path, dpi=300)
        #cv2.waitKey(0)
        #plt.clf()
       

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

    path_to_file = os.path.join(
    cfg.DATA.PATH_TO_DATA_DIR, "{}.txt".format("test")
    )
    assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)
    _path_to_seq_imgs  = []
    with open(path_to_file, "r") as f:
        for clip_idx, path_label in enumerate(f.read().splitlines()):
            if len(path_label.split()) == 2 :
                path, label = path_label.split()
                name=os.path.basename(path)
            
                _path_to_seq_imgs.append(
                    os.path.join(name)
                )
                # print("dir name = {}".format(name))

    print("size of _path_to_seq_imgs = {}".format(len(_path_to_seq_imgs)))
    cam_view_test(test_loader, model,  cfg,_path_to_seq_imgs)