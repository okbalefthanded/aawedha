# Adadpted from: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/conv/DynamicConv.py
# Implements: Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., &#38; Liu, Z. (2020). Dynamic convolution: 
# Attention over convolution kernels. Proceedings of the IEEE Computer Society Conference on 
# Computer Vision and Pattern Recognition, 11027–11036. https://doi.org/10.1109/CVPR42600.2020.01104
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30,init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes>=ratio
        hidden_planes = in_planes//ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att = self.avgpool(x) #bs, dim, 1, 1
        att = self.net(att).view(x.shape[0], -1) #bs, K
        return F.softmax(att / self.temprature, -1)

class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,
                 dilation=1, grounps=1, bias=True ,K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size =_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                 init_weight=init_weight)
        
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//grounps, ks1, ks2), requires_grad=True)
        if(bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化 (initialization)
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x) #bs, K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1) #K, -1
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]
        aggregate_weight = torch.mm(softmax_att, weight).view(bs*self.out_planes, 
                                                              self.in_planes//self.groups,
                                                              ks1, ks2) #bs*out_p, in_p, k, k

        if(self.bias is not None):
            bias = self.bias.view(self.K, -1) #K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1) #bs, out_p
            output=F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, 
                            stride=self.stride, padding=self.padding, 
                            groups=self.groups*bs,
                            dilation=self.dilation)
        else:
            output=F.conv2d(x, weight=aggregate_weight,
                            bias=None, stride=self.stride,
                            padding=self.padding,
                            groups=self.groups*bs,
                            dilation=self.dilation)

        _, out, h1, w1 = output.shape
        # output=output.view(bs,self.out_planes,h,w)
        output=output.view(bs, self.out_planes, h1, w1)
        return output    



class Attention1d(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.temprature = temprature
        assert in_planes>=ratio
        hidden_planes = in_planes//ratio
        self.net = nn.Sequential(
            nn.Conv1d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att = self.avgpool(x) #bs, dim, 1, 1
        att = self.net(att).view(x.shape[0], -1) #bs, K
        return F.softmax(att / self.temprature, -1)

class DynamicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,
                 dilation=1, grounps=1, bias=True ,K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes   = in_planes
        self.out_planes  = out_planes
        self.kernel_size = kernel_size
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups   = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention1d(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                 init_weight=init_weight)
        
        ks1 = self.kernel_size

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//grounps, ks1), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias=None
        
        if self.init_weight:
            self._initialize_weights()

        #TODO 初始化 (initialization)
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs, in_planels, f = x.shape
        softmax_att = self.attention(x) #bs, K
        x = x.view(1, -1, f)
        weight = self.weight.view(self.K, -1) #K, -1
        ks1 = self.kernel_size
        aggregate_weight = torch.mm(softmax_att, weight).view(bs*self.out_planes, 
                                                              self.in_planes//self.groups,
                                                              ks1) #bs*out_p, in_p, k, k

        if self.bias is not None:
            bias = self.bias.view(self.K, -1) #K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1) #bs, out_p
            output=F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, 
                            stride=self.stride, padding=self.padding, 
                            groups=self.groups*bs,
                            dilation=self.dilation)
        else:
            output=F.conv1d(x, weight=aggregate_weight,
                            bias=None, stride=self.stride,
                            padding=self.padding,
                            groups=self.groups*bs,
                            dilation=self.dilation)

        _, out, f1 = output.shape
        # output=output.view(bs,self.out_planes,h,w)
        output=output.view(bs, self.out_planes, f1)
        return output

    