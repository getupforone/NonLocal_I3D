import torch
import torch.nn as nn


class Nonlocal(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        norm_type="batchnorm",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.norm_type = norm_type
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm)
    
    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm):
        self.conv_theta = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1,padding=0
        )
        self.conv_out = nn.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_out.zero_init = zero_init_final_conv

        if self.norm_type == "batchnorm":
            self.bn = nn.BatchNorm3d(
                self.dim, eps=self.norm_eps, momentum=self.norm_momentum
            )
            self.bn.trnsform_final_bn = zero_init_final_norm
        elif self.norm_type == "layernorm":
            self.ln = nn.GroupNorm(1, self.dim, eps=self.norm_eps, affine=False)
        elif self.norm_type=="none":
            pass
        else:
            raise NotImplementedError(
                "Norm type {} is not supported".format(self.norm_type)
            )
        
        if self.use_pool:
            self.pool = nn.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0,0,0],
            )
    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

        # A subsampling trick can be used to futher reduce computation.
        # We perform this in the spatial domain. which  can reduce the amount of
        # pairwise computation by 1/4.
        # This Can be done by adding a max pooling layer after pi and g 
        # but this trick does not change the image dimension.
        #theta(THWX512) x pi(T'H'W'x512) = THW xT'H'W'
        #thetapi(THW xT'H'W') xg(T'H'W'x 512) = THWx512
        if self.use_pool:
            x = self.pool(x)
        
        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, TxHxW) * (N, C, TxHxW) => (N, TxHxW, TxHxW)
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))

        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim = 2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )
        
        # (N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ntg,ncg->nct",(theta_phi, g))

        #(N, C, TxHxW) => (N, C, T, H, W)
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        p = self.conv_out(theta_phi_g)
        if self.norm_type == "batchnorm":
            p = self.bn(p)
        elif self.norm_type == "layernorm":
            p = self.ln(p)
        return x_identity + p
