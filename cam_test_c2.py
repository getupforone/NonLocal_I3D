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
import re

logger = logging.get_logger(__name__)
TARGET_DIR = "."
RESULT_IMG_DIR = "./img2"
RESULT_MULTI_IMG_DIR = "./multi_img2"
import multiprocessing as mp
def upsample(in_img, in_scale_factor= (2,32,32)):
    upsample = nn.Upsample( scale_factor = in_scale_factor, mode = "bilinear")
    return upsample

def cam_view_test_c2(test_loader, model, cfg, path_to_seq_imgs, cur_epoch):
    model.eval()
    
    for cur_iter, (inputs, labels, video_idx) in enumerate(test_loader):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        # print("cur_iter :{}".format(cur_iter))
        # print("video_idx :{}".format(video_idx.item()))
        # # preds : (8, 2, 1, 1, 1)
        # # feat : (8, 2048, 4, 7, 7)
        # # fc_w : (2, 2048)
        # print("input shape = {}".format(inputs.shape))
        preds,feat,fc_w = model(inputs) # y, x, w
        print("preds shape = {}".format(preds.shape))
        print("feat shape = {}".format(feat.shape))
        print("fc_w shape = {}".format(fc_w.shape))
        preds = torch.squeeze(preds)
        print(">>> preds is {}".format(preds))
        #preds_idx = torch.argmax(preds, dim=1)
        preds_idx = torch.argmax(preds)
        # print("preds_idx shape = {}".format(preds_idx.shape))
        # print("preds_idx  = {}".format(preds_idx))
        # # fc_w_wide : (8, 2048 )
        fc_w_wide = fc_w[preds_idx] # True label weight. we'd better plot false label case.
        pred = preds[preds_idx]

        # del preds_idx
        # print("fc_w_wide shape = {}".format(fc_w_wide.shape))
        # # feat : (8, 2048, 4, 7, 7)
        # # feat_per : (8, 4, 7, 7, 2048)
        feat_per = feat.permute((0,2,3,4,1))
        print("feat_per shape = {}".format(feat_per.shape))
        feat_per = torch.squeeze(feat_per)
        print("feat_per shape = {}".format(feat_per.shape))
        # ouput : ( 4, 7, 7, 2048) * (2048) = (4,7,7)
        output = torch.matmul(feat_per,fc_w_wide)
        print("output shape = {}".format(output.shape))


        del preds,feat,fc_w 
        del fc_w_wide
        del feat_per
        # print("output shape = {}".format(output.shape))

        output_unsqz =torch.unsqueeze(output,0) 
        print("output_unsqz shape = {}".format(output_unsqz.shape))
        output_unsqz =torch.unsqueeze(output_unsqz,0)
        print("output_unsqz shape = {}".format(output_unsqz.shape))
        # print(output_unsqz.shape)

        if output.shape[0] != cfg.DATA.NUM_FRAMES :
            scale_num = cfg.DATA.NUM_FRAMES//output.shape[0]
            upsample2=nn.Upsample(scale_factor = (scale_num,1,1), mode = 'trilinear')
            up_img_temp = upsample2(output_unsqz)
            # up_img_temp : (8,7,7)
            #up_img_temp = upsample2(output)
            print(up_img_temp.shape)
        else:
            up_img_temp = output_unsqz
        del output
        del output_unsqz

        upsample=nn.Upsample(scale_factor = (32,32), mode = 'bilinear')
        up_img_temp_sqz = torch.squeeze(up_img_temp,0)
        del up_img_temp
        print(up_img_temp_sqz.shape)
        up_img = upsample(up_img_temp_sqz)
        del up_img_temp_sqz        
        print(up_img.shape)
        up_img = torch.squeeze(up_img,0)
        print("up_img shape = {}".format(up_img.shape))
        # print(up_img)
       
       
        up_img_num = up_img.shape[0]
        v_idx = video_idx.item()
        del video_idx
        #p_idx = (int)(v_idx/(cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)) 
        p_idx = v_idx # we did not spatial clop for the test session

        print("v_idx = {}/ p_idx = {}/ path_to_seq_imgs size = {}/size of DataLoader = {}".format(v_idx,p_idx, len(path_to_seq_imgs),len(test_loader)))
        shot_num = path_to_seq_imgs[p_idx]
        print("shot_num ={}".format(shot_num))

        img_dir = os.path.join(RESULT_IMG_DIR)
        if os.path.exists(img_dir):
            print("dir {} exists".format(img_dir))
        else:
            os.mkdir(img_dir)
        img_dir2= os.path.join(RESULT_IMG_DIR,shot_num)
        if os.path.exists(img_dir):
            print("dir {} exists".format(img_dir2))
        else:
            os.mkdir(img_dir2)

        # multi_img_dir = os.path.join(RESULT_MULTI_IMG_DIR,shot_num)
        multi_img_dir = os.path.join(RESULT_MULTI_IMG_DIR)
        if os.path.exists(multi_img_dir):
            print("dir {} exists".format(multi_img_dir))
        else:
            os.mkdir(multi_img_dir)
        print("shape of inputs = {}".format(inputs.shape))


        inputs_sq = torch.squeeze(inputs,0)
        del inputs
        inputs_per = inputs_sq.permute((1,2,3,0))
        del inputs_sq
        inputs_per = (inputs_per.cpu()).detach() * torch.tensor(cfg.DATA.STD)

        inputs_per = inputs_per + torch.tensor(cfg.DATA.MEAN)

        # inputs_per = inputs_per * 255.0
        # input_img = ((inputs_per.cpu()).detach()).numpy()
        input_img = inputs_per.numpy()
        del inputs_per
        result_np_img = ((up_img.cpu()).detach()).numpy()
        del up_img
        labels  = ((labels.cpu()).detach()).numpy()
       
        
        print("result_np_img type is {} dtype is {}".format(type(result_np_img), result_np_img.dtype))
        fig, ax = plt.subplots()
        #fig = plt.figure()
        if labels[0] == 1:
            img_label = 'True'
        elif labels[0] == 0:
            img_label = 'False'
        elif labels[0] == 2:
            img_label = 'Disrupt'
        del labels
        fig_title = "kstartv_i3d_nln_10_12_2_224_{}_{}_{}_{}_{}".format(img_label,shot_num,cur_epoch,pred,preds_idx)
        del pred
        del preds_idx 
        plt.rcParams['figure.figsize'] = [16, 16]
        plt.title(fig_title)
        plt.axis('off')
        #up_img_num = num of batch 4,6,8,10,12
        disp_img_num = up_img_num*3
        rows = 0
        cols = 4
        if disp_img_num%cols !=0:
            rows = int(disp_img_num/cols) + 1
        else:
            rows = int(disp_img_num/cols)
        print("rows/cols = ({}/{})".format(rows,cols))

        xlabels = ["xlabel", "(t0) img","(t1)img","(t2)img","(t3)img","(t4)img","(t5)img","(t6)img","(t7)img","(t8)img","(t9)img","(t10)img","(t11)img"]
        xlabels2 = ["xlabel2", "(t0)cam","(t1)cam","(t2)cam","(t3)cam","(t4)cam","(t5)cam","(t6)cam","(t7)cam","(t8)cam","(t9)cam","(t10)cam","(t11)cam"]


        # input_img = input_img - np.min(input_img)
        # input_img = input_img/np.max(input_img)
        # input_img = (input_img*255).astype(np.uint8)


        # frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        # frames = frames / torch.tensor(self.cfg.DATA.STD)
        for img_idx in range(up_img_num):
            # img_name = "{}_{}_{}_{}_{}.jpg".format(shot_num,v_idx,cur_iter,img_idx,cur_epoch)
            # norm_img_name = "{}_{}_{}_{}_{}_norm.jpg".format(shot_num,v_idx,cur_iter,img_idx,cur_epoch)           
            
            # img_path = os.path.join(img_dir,img_name)
            # print(img_path)

            # # print("img_path : {}".format(img_path))
            # norm_img_path = os.path.join(img_dir,norm_img_name)
            
            # #im_gray = (result_np_img[img_idx]).astype('uint8')
            # im_gray = (result_np_img[img_idx])            
            # norm_img = (cv2.normalize(im_gray,None,0,255,norm_type=cv2.NORM_MINMAX)).astype('uint8')
            # #im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
            # im_color = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)                        
            # cv2.imwrite(img_path,im_color)
            # cv2.imwrite(norm_img_path,norm_img)

            #img_idx 0~7            
            xlab_idx = img_idx % up_img_num +1     #xlab_idx 1~up_img_num               
            ax = fig.add_subplot(rows, cols, img_idx + 1)
            
            ax.imshow(input_img[img_idx]) 
            ax.set_xlabel(xlabels[xlab_idx])
            ax.set_xticks([]), ax.set_yticks([])  
            # del im_gray
            # del norm_img
            # del im_color
            del ax
        
        max_lim = 150
        min_lim = 0
        result_np_img = result_np_img - np.min(result_np_img)
        result_np_img = result_np_img/np.max(result_np_img)
        maxv = np.max(result_np_img)
        minv = np.min(result_np_img)
        max_lim = 1
        min_lim = 0
        print("maxv and minv = {}/{}".format(maxv,minv))
        for img_idx in range(up_img_num):            
            sub_img_idx = img_idx + up_img_num #8~15    
            xlab_idx = img_idx % up_img_num +1     #xlab_idx 1~up_img_num   
            r_img = result_np_img[img_idx]
            ax = fig.add_subplot(rows, cols, sub_img_idx + 1)

            ax.imshow(input_img[img_idx], alpha=0.7)
            # print(input_img[img_idx])
            # print( "input_img range is ({}~{})".format(np.amin(input_img[img_idx]),np.amax(input_img[img_idx])) )
            ax.imshow(r_img, cmap='jet', vmin= min_lim, vmax = max_lim, alpha=0.4)
            ax.set_xlabel(xlabels2[xlab_idx])                      
            ax.set_xticks([]), ax.set_yticks([]) 

            
            # print(r_img)
            # print( "r_img range is ({}~{})".format(np.amin(r_img),np.amax(r_img)) )
            del r_img  


        for img_idx in range(up_img_num):            
            sub_img_idx = img_idx + up_img_num*2 #8~15    
            xlab_idx = img_idx % up_img_num +1     #xlab_idx 1~up_img_num   
            
            ax = fig.add_subplot(rows, cols, sub_img_idx + 1)
            
            r_img = result_np_img[img_idx]
            ax.imshow(input_img[img_idx], alpha=0.7)
            ax.imshow(r_img, cmap='jet', vmin=min_lim, vmax = max_lim, alpha=0.3)
            heatmap = ax.pcolor(r_img,cmap='jet', vmin=min_lim, vmax = max_lim)
            cbar = plt.colorbar(heatmap)
            ax.imshow(r_img, cmap='jet', vmin=min_lim, vmax = max_lim, alpha=0.4)
            ax.set_xlabel(xlabels2[xlab_idx])                      
            ax.set_xticks([]), ax.set_yticks([]) 
            del r_img   
            
            # r_img = -1.0 * result_np_img[0]   
            # heatmap = ax.pcolor(r_img,cmap='jet', vmin=min_lim, vmax = max_lim)
            # cbar = plt.colorbar(heatmap)
            del heatmap
            del cbar
            # del ax
        # cbar.set_label('Color Intensity')

            # ax.set_title(img_name) 
        #plt.show()
        multi_img_title = "{}.jpg".format(fig_title)
        multi_img_path = os.path.join(multi_img_dir,multi_img_title)
        print("multi_img_path = {}".format(multi_img_path))
        plt.savefig(multi_img_path, dpi=300)
        #cv2.waitKey(0)
        plt.clf()
        # del preds,feat,fc_w
        del input_img
        del result_np_img        
        del fig, ax
       

def cam_test_c2(cfg):
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
        del last_checkpoint
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # assert(
    #     len(test_loader.dataset)
    #     % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
    #     == 0
    # )
    mode = "test"
    path_to_dir = os.path.join(
        # self.cfg.DATA.PATH_TO_DATA_DIR, "{}.txt".format(self.mode)
        cfg.DATA.PATH_TO_DATA_DIR, "{}".format(mode)
    )
    print("KstarTV::path_to_dir is {}".format(path_to_dir))
    assert os.path.exists(path_to_dir), "{} dir not found".format(path_to_dir)
    file_path_list = []
    file_name_list = np.sort(os.listdir(path_to_dir))
    _path_to_seq_imgs  = []
    for f_name in file_name_list:
        
        f_name_split = re.split("[._]",f_name)
        # print("f_name = {}".format(f_name))
        
        # print("f_name_split = {}".format(f_name_split))
        _path_to_seq_imgs.append(f_name_split[0])
    # path_to_file = os.path.join(
    # cfg.DATA.PATH_TO_DATA_DIR, "{}.txt".format("test")
    # )
    # assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)
    # _path_to_seq_imgs  = []
    # with open(path_to_file, "r") as f:
    #     for clip_idx, path_label in enumerate(f.read().splitlines()):
    #         if len(path_label.split()) == 2 :
    #             path, label = path_label.split()
    #             name=os.path.basename(path)
    #             del path, label    
    #             _path_to_seq_imgs.append(
    #                 os.path.join(name)
    #             )
    #             del name
    #             # print("dir name = {}".format(name))


    
    print("size of _path_to_seq_imgs = {}".format(len(_path_to_seq_imgs)))

    start_epoch = 0
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # loader.shuffle_dataset(test_loader, cur_epoch)
        cam_view_test_c2(test_loader, model,  cfg,_path_to_seq_imgs,cur_epoch)
