import torch
import torch.nn as nn

import utils.weight_init_helper as init_helper
from models import head_helper, resnet_helper, stem_helper

_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

#Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],      #conv1 temporal kernel.
        [[3]],      #res2 temporal kernel.
        [[3, 1]],   #res3 temporal kernel.
        [[3, 1]],   #res4 temporal kernel.
        [[1, 3]],   #res5 temporal kernel.
    ],

    "slowonly": [
        [[1]],      #conv1 temporal kernel.
        [[1]],      #res2 temporal kernel.
        [[1]],   #res3 temporal kernel.
        [[3]],   #res4 temporal kernel.
        [[3]],   #res5 temporal kernel.
    ],
}
_POOL1 = {
    "c2d": [[2, 1, 1]],
    #"i3d": [[2, 1, 1]],
    "i3d": [[1, 1, 1]], 
    "slowonly": [[1, 1, 1]],
}

class ResNetModel(nn.Module):
    def __init__(self, cfg):
        super(ResNetModel, self).__init__()
        self._construct_network(cfg)
        init_helper.init_weight(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        
    
    def _construct_network(self, cfg):
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        self.isCAMTest = cfg.TEST.IS_CAM_TEST
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        print("arch : {}".format(cfg.MODEL.ARCH))
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]


        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
        )
        pool = nn.MaxPool3d(
            kernel_size=pool_size[0], stride=pool_size[0],padding=[0,0,0],
            #kernel_size=[1,3,3], stride=pool_size[0],padding=[0,0,0],
        )
        self.add_module("pool", pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
        )
        
        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
        )
        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                cfg.DATA.NUM_FRAMES // pool_size[0][0],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func="softmax",
            is_cam_test = cfg.TEST.IS_CAM_TEST
        )
        # print("pool_size z! : {}".format(pool_size))
        # print("pool_size[0][1] z! : {}".format(pool_size[0][1]))
        # print(" cfg.DATA.CROP_SIZE z! : {}".format( cfg.DATA.CROP_SIZE))
        # print("pool_size zz : {}".format([
        #         cfg.DATA.NUM_FRAMES // pool_size[0][0],
        #         cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
        #         cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
        #     ]))
        # print("\n=======================================")
        # print("resnet : NUM_CLASSES ={}".format(cfg.MODEL.NUM_CLASSES))
        # print("=======================================\n")
        

    def forward(self, x):
        #print("s0 :x dim  = {}".format(x.shape))
        x = self.s1(x)
        #print("s1 :x dim  = {}".format(x.shape))
        x = self.s2(x)
        #print("s2 :x dim  = {}".format(x.shape))
        pool = getattr(self, "pool")
        x = pool(x)
        #print("pool :x dim  = {}".format(x.shape))
        x = self.s3(x)
        #print("s3 :x dim  = {}".format(x.shape))
        x = self.s4(x)
        #print("s4 :x dim  = {}".format(x.shape))
        x = self.s5(x)
        #print("s5 :x dim  = {}".format(x.shape))
        if self.isCAMTest == True:
            x,feat,fc_w = self.head(x)
            return x,feat,fc_w
        else :
            x = self.head(x)
        #print("head :x dim  = {}".format(x.shape))
        
        return x