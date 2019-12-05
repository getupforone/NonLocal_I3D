import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill

def init_weight(model, fc_init_std=0.01, zero_init_final_bn=True):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if(
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0

            m.weight.data.fill_(batchnorm_weight)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()