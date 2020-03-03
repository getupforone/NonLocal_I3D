import torch
import torch.nn as nn

class ResNetBasicHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        super(ResNetBasicHead, self).__init__()
        # assert(
        #     len({len(pool_size), len(dim_in)})== 1
        # ), "pathway dimensions are not consistent."
        #print("!!pool_size = {}".format(pool_size))
        avg_pool = nn.AvgPool3d(pool_size, stride=1)
        self.add_module("avgpool", avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
    def forward(self, inputs):
        #pool_out = []
        m = getattr(self, "avgpool")
        print("head : dim-1 = {}".format(inputs.shape))
        #pool_out.append(m(inputs))
        #x = torch.cat(pool_out, 1)
        x = m(inputs)
        print("head : dim0 = {}".format(x.shape))
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        fc_w = self.projection.weight
        print("fc_w dim = {}".format(fc_w.shape))
        print("fc_w[0] = {}".format(fc_w[0]))

        print("head : dim1 = {}".format(x.shape))
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

    
        #x = self.act(x)
        #x = x.mean([1, 2, 3])
        print("head : dim2 = {}".format(x.shape))
        #print("x val1 = {}".format(x))
        x = x.view(x.shape[0], -1)
        #print("x val2 = {}".format(x))
        
        print("head : dim3 = {}".format(x.shape))

        return x,input,fc_w