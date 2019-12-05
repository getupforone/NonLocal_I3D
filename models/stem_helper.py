#!/usr/bin/env python3

import torch.nn as nn

class VideoModelStem(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(VideoModelStem, self).__init__()
        assert(
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            ) == 1
        ),"Input dimensions are not consistent."
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt

        self._construct_stem(dim_in, dim_out)
    def _construct_stem(self, dim_in, dim_out):
        stem = ResNetBasicStem(
            dim_in,
            dim_out,
            self.kernel,
            self.stride,
            self.padding,
            self.inplace_relu,
            self.eps,
            self.bn_mmt,
        )
        self.add_module("stem",stem)
    def forward(self, x):
        m = getattr(self,"stem")
        x = m(x)
        return x

class ResNetBasicStem(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1,3,3], stride=[1,2,2], padding=[0,1,1]
        )
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x