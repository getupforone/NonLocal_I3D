import torch.nn as nn

from models.nonlocal_helper import Nonlocal

def get_trans_func(name):
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert(
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)

    return trans_funcs[name]

class BasicTransform(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        # conv3d(Tx3x3), BN, ReLU
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3,3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size //2), 1, 1],
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(
            dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # conv3d(1x3x3), BN.
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1,3,3],
            stride=[1,1,1],
            padding=[0,1,1],
            bias=False,
        )
        self.b_bn = nn.BatchNorm3d(
            dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_bn.transform_final_bn = True
    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x

class BottleneckTransform(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups)
    
    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(
            dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, 1, 1],
            groups=num_groups,
            bias=False,
        )
        self.b_bn = nn.BatchNorm3d(
            dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = nn.BatchNorm3d(
            dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        x = self.c(x)
        x = self.c_bn(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
        )
    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
    ):
        # Use skip connection with projection if dim or res change
        if (dim_in != dim_out) or (stride != 1) :
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
            )
            self.branch1_bn = nn.BatchNorm3d(
                dim_out, eps=self._eps, momentum=self._bn_mmt
            )

        self.barnch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x

class ResStage(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
    ):
        super(ResStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self.temp_kernel_size = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            ) == 1
        )
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            instantiation,
        )
    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        instantiation,
    ):
        for i in range(self.num_blocks):
            trans_func = get_trans_func(trans_func_name)
            res_block = ResBlock(
                dim_in if i == 0 else dim_out,
                dim_out,
                self.temp_kernel_size[i],
                stride if i == 0 else 1,
                trans_func,
                dim_inner,
                num_groups,
                stride_1x1=stride_1x1,
                inplace_relu=inplace_relu,
            )
            self.add_module("res{}".format(i), res_block)
            if i in nonlocal_inds:
                nln = Nonlocal(
                    dim_out,
                    dim_out // 2,
                    [1, 2, 2],
                    instantiation=instantiation,
                )
                self.add_module(
                    "nonlocal{}".format(i), nln
                )
    def forward(self, inputs):
        output = []
        x = inputs
        for i in range(self.num_block):
            m = getattr(self, "res{}".format(i))
            x = m(x)
            if hasattr(self, "nonlocal{}".format(i)):
                nln = getattr("nonlocal{}".format(i))
                b, c, t, h, w = x.shape
                if self.nonlocal_group > 1:
                    # Fold temporal dimension into batch dimension.
                    #( b,c,t,h,w) => (b,t,c,h,w)
                    x = x.permute(0, 2, 1, 3, 4)
                    x = x.reshape(
                        b * self.nonlocal_group,
                        t // self.nonlocal_group,
                        c,
                        h,
                        w,
                    )
                    x = x.permute(0,2,1,3,4)
                x = nln(x)
                if self.nonlocal_group > 1:
                    x = x.permute(0,2,1,3,4)
                    x = x.reshape(b,t,c,h,w)
                    x = x.permute(0,2,1,3,4)
            output.append(x)

        return output
        