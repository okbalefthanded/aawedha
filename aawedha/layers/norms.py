import torch.nn.functional as F
import numpy as np
import torch
import math
import torch.nn as nn

# https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

#  https://github.com/ShangHua-Gao/RBN
class RepresentativeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(RepresentativeBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        ### weights for affine transformation in BatchNorm ###
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.weight.data.fill_(1)
            self.bias.data.fill_(0)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        ### weights for centering calibration ###
        self.center_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.center_weight.data.fill_(0)
        ### weights for scaling calibration ###
        self.scale_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_weight.data.fill_(0)
        self.scale_bias.data.fill_(1)
        ### calculate statistics ###
        self.stas = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        self._check_input_dim(input)
        center = self.center_weight.view(1,self.num_features,1,1)
        ####### centering calibration begin #######
        tmp = center*self.stas(input)
        tmp2 = input
        input =  tmp + tmp2
        ####### centering calibration end #######

        ####### BatchNorm begin #######
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        output = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        ####### BatchNorm end #######

        ####### scaling calibration begin #######
        scale_factor = torch.sigmoid(self.scale_weight*self.stas(output)+self.scale_bias)
        ####### scaling calibration end #######
        if self.affine:
            return self.weight*scale_factor*output + self.bias
        else:
            return scale_factor*output