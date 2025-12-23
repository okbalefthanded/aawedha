import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple


def cosine_rampdown(current, rampdown_length):
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

class PreConv_2(_ConvNd):
    """
    Preconditioned convolution
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=False, padding_mode='zeros', affine = True,
        bn = True, momentum = None, track_running_stats=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        # padding = _pair(padding)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        super(PreConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.affine = affine
        self.bn = bn
        self.register_buffer('running_V', torch.zeros(1))
        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0
        self.momentum = momentum
        k = kernel_size[0]
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(out_channels, affine = affine)
        if affine:
            self.bpconv = nn.Conv2d(out_channels, out_channels, k, padding="valid", groups=out_channels, bias=True)
        else:
            self.bpconv = nn.Sequential()
    
    def _truncate_circ_to_cross(self, out):
        # First calculate how much to truncate
        if isinstance(self.padding, str):
            padd = _pair(0)
        else:
            padd = self.padding
        out_sizex_start = self.kernel_size[0] - 1 - padd[0] # self.padding[0]
        out_sizey_start = self.kernel_size[1] - 1 - padd[1] # self.padding[1]
       
        if out_sizex_start != 0 and out_sizey_start != 0:
            out = out[:, :, out_sizex_start: -out_sizex_start, out_sizey_start:-out_sizey_start]
        elif out_sizex_start == 0:
            if out_sizey_start != 0:
                out = out[:, :, :, out_sizey_start:-out_sizey_start]
        elif out_sizey_start == 0:
            if out_sizex_start != 0:
                out = out[:, :, out_sizex_start: -out_sizex_start, :]
        # Also considering stride
        if self.stride[0] > 1:
            out = out[..., ::self.stride[0], ::self.stride[1]]
        return out
    
    def _calculate_running_estimate(self, current_V):
        with torch.no_grad():
            exponential_average_factor = 0.0
            if self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = max(cosine_rampdown(self.num_batches_tracked,40000),1e-2)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
            self.running_V = exponential_average_factor * current_V\
                            + (1 - exponential_average_factor) * self.running_V
        
    def conv2d_forward(self, input, weight, stride=1):
        # padding should be kernel size - 1
        # padd = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        # return F.conv2d(input, weight, self.bias, stride,
        #                 padd, self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, stride,
                         self.padding, self.dilation, self.groups)
    
    
    def preconditioning(self, cout, kernel):
        if (self.kernel_size[0]==1) or (self.kernel_size[1] ==1):
            V = kernel ** 2
            V = torch.sum(V, dim=1)
            V = torch.exp(-0.5*torch.log(V))
            with torch.no_grad():
                if self.training:
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            return V*cout 
        else:
            final_size_x = cout.size(-2)
            final_size_y = cout.size(-1)
            f_input = torch.fft.rfft2(cout)
            with torch.no_grad():
                if self.training:
                    pad_kernel = F.pad(kernel, [0, final_size_y-self.kernel_size[1], 0, final_size_x-self.kernel_size[0]])
                    f_weight = torch.fft.rfft2(pad_kernel)
                    V = f_weight.abs() ** 2 # f_weight is a complex number 
                    V = torch.sum(V, dim=1)
                    V = torch.exp(-0.5*torch.log(V))
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            output = torch.fft.irfft2(f_input.mul_(V))
            return output
        
    def forward(self, input):
        c_out = self.conv2d_forward(input, self.weight)
        p_out = self.preconditioning(c_out, self.weight.data.detach())
        # Truncate the preconditioning result for desired spatial size and stride
        if (input.shape[-1] != c_out.shape[-1]) and (input.shape[-2] != c_out.shape[-2]):
            p_out = self._truncate_circ_to_cross(p_out)
        # Affine part
        output =  self.bpconv(p_out)
        # If use BatchNorm
        if self.bn:
            output = self.batch_norm(output)
        return output


class PreConv(_ConvNd):
    """
    Preconditioned convolution
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=False, padding_mode='zeros', affine = True,
        bn = True, momentum = None, track_running_stats=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        # padding = _pair(padding)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        super(PreConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        ##### Change the default padding for "valid" / "same" #####
        ##### Need to consider even or odd kernel size here #####
        if padding == "valid":
            self.padding = (0,0)
            self.paddd = []
        elif padding == "same":
            self.paddd = [self._calculate_padding(self.kernel_size[0]), self._calculate_padding(self.kernel_size[1])]
        else:
            self.paddd = []
        self.affine = affine
        self.bn = bn
        self.register_buffer('running_V', torch.zeros(1))
        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0
        self.momentum = momentum
        k1,k2 = kernel_size
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(out_channels, affine = affine)
        if affine:
            ##### Change padding here #####
            self.bpconv = nn.Conv2d(out_channels, out_channels, (k1,k2), padding = "same", groups=out_channels, bias=True)
        else:
            self.bpconv = nn.Sequential()

    def _calculate_padding(self, ks):
        # ks: @int, kernel size
        if ks % 2 == 1: # Odd
            paddd = [int(ks // 2), int(ks // 2)]
        else: # even
            paddd = [int(ks // 2 - 1), int(ks // 2)]
        return paddd

    def _truncate_circ_to_cross(self, out):
        # First calculate how much to truncate
        if len(self.paddd) != 0: # Means the same case
            paddd_0, paddd_1 = self.paddd[0], self.paddd[1] # two list
            out_sizex_start = self.kernel_size[0] - 1 - paddd_0[0]
            out_sizex_end = self.kernel_size[0] - 1 - paddd_0[1]
            out_sizey_start = self.kernel_size[1] - 1 - paddd_1[0]
            out_sizey_end = self.kernel_size[1] - 1 - paddd_1[1]
        else:
            out_sizex_start = self.kernel_size[0] - 1 - self.padding[0]
            out_sizex_end = self.kernel_size[0] - 1 - self.padding[0]
            out_sizey_start = self.kernel_size[1] - 1 - self.padding[1]
            out_sizey_end = self.kernel_size[1] - 1 - self.padding[1]
        # print("size need to crop,", out_sizex_start,out_sizex_end,out_sizey_start,out_sizey_end)
        if out_sizex_start != 0 and out_sizey_start != 0:
            out = out[:, :, out_sizex_start: -out_sizex_end, out_sizey_start:-out_sizey_end]
        elif out_sizex_start == 0:
            if out_sizey_start != 0:
                out = out[:, :, :, out_sizey_start:-out_sizey_end]
        elif out_sizey_start == 0:
            if out_sizex_start != 0:
                out = out[:, :, out_sizex_start: -out_sizex_end, :]
        # Also considering stride
        if self.stride[0] > 1:
            out = out[..., ::self.stride[0], ::self.stride[1]]
        return out
    
    def _calculate_running_estimate(self, current_V):
        with torch.no_grad():
            exponential_average_factor = 0.0
            if self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = max(cosine_rampdown(self.num_batches_tracked,40000),1e-2)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
            self.running_V = exponential_average_factor * current_V\
                            + (1 - exponential_average_factor) * self.running_V
        
    def conv2d_forward(self, input, weight, stride=1):
        # padding should be kernel size - 1
        padd = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        return F.conv2d(input, weight, self.bias, stride,
                        padd, self.dilation, self.groups)
    
    def preconditioning(self, cout, kernel):
        # print("cout shape,", cout.shape)
        #if (self.kernel_size[0]==1) or (self.kernel_size[1] ==1):
        if (self.kernel_size[0]==1) and (self.kernel_size[1] ==1):
            V = kernel ** 2
            V = torch.sum(V, dim=1)
            V = torch.exp(-0.5*torch.log(V))
            with torch.no_grad():
                if self.training:
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            return V*cout 
        else:
            final_size_x = cout.size(-2)
            final_size_y = cout.size(-1)
            f_input = torch.fft.rfft2(cout)
            with torch.no_grad():
                if self.training:
                    pad_kernel = F.pad(kernel, [0, final_size_y-self.kernel_size[1], 0, final_size_x-self.kernel_size[0]])
                    f_weight = torch.fft.rfft2(pad_kernel)
                    V = f_weight.abs() ** 2 # f_weight is a complex number 
                    V = torch.sum(V, dim=1)
                    V = torch.exp(-0.5*torch.log(V))
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            output = torch.fft.irfft2(f_input.mul_(V), s=(final_size_x,final_size_y))
            return output
        
    def forward(self, input):
        c_out = self.conv2d_forward(input, self.weight)
        p_out = self.preconditioning(c_out, self.weight.data.detach())
        # print("out after prec size,", p_out.shape)
        # Truncate the preconditioning result for desired spatial size and stride
        p_out = self._truncate_circ_to_cross(p_out)
        # Affine part
        output =  self.bpconv(p_out)
        # If use BatchNorm
        if self.bn:
            output = self.batch_norm(output)
        return output


class Preconditioned_Conv2d(PreConv):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))