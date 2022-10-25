# https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/conv/CondConv.py
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint

class Attention(nn.Module):
    def __init__(self,in_channels, num_experts, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net     = nn.Conv2d(in_channels, num_experts, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if(init_weight):
            self._initialize_weights()

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
        att = self.avgpool(x) #bs,dim,1,1
        att = self.net(att).view(x.shape[0],-1) #bs, K
        return self.sigmoid(att)

class CondConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, num_experts=4, 
                 init_weight=True):
        super().__init__()
        self.in_planes=in_channels
        self.out_planes=out_channels
        self.kernel_size= _pair(kernel_size)
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        self.K=num_experts
        self.init_weight=init_weight
        self.attention=Attention(in_channels=in_channels, num_experts=num_experts,
                                 init_weight=init_weight)       
        
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]

        self.weight=nn.Parameter(torch.randn(
                                num_experts, out_channels, in_channels//groups,
                                ks1, ks2),
                                requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(num_experts, out_channels),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x) #bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K,-1) #K,-1
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]
        aggregate_weight=torch.mm(softmax_att, weight).view(bs*self.out_planes,
                                                           self.in_planes//self.groups,
                                                           ks1,
                                                           ks2) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                            stride=self.stride, padding=self.padding,
                            groups=self.groups*bs,
                            dilation=self.dilation)
        else:
            output=F.conv2d(x, weight=aggregate_weight, bias=None, 
                            stride=self.stride, padding=self.padding, 
                            groups=self.groups*bs, 
                            dilation=self.dilation)
        
        n, c, h1, w1 = output.shape
        # output=output.view(bs, self.out_planes, h, w)
        output = output.view(bs, self.out_planes, h1, w1)
        return output


class AttentionConstraint(nn.Module):
    def __init__(self,in_channels, num_experts, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net     = Conv2dWithConstraint(in_channels, num_experts, kernel_size=1, bias=False, max_norm=1)             
        self.sigmoid = nn.Sigmoid()

        if(init_weight):
            self._initialize_weights()

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
        att = self.avgpool(x) #bs,dim,1,1
        att = self.net(att).view(x.shape[0],-1) #bs, K
        return self.sigmoid(att)

class CondConvConstraint(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, num_experts=4, 
                 init_weight=True):
        super().__init__()
        self.in_planes=in_channels
        self.out_planes=out_channels
        self.kernel_size= _pair(kernel_size)
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        self.K=num_experts
        self.init_weight=init_weight
        self.attention=AttentionConstraint(in_channels=in_channels, num_experts=num_experts,
                                 init_weight=init_weight)       
        
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]

        self.weight=nn.Parameter(torch.randn(
                                num_experts, out_channels, in_channels//groups,
                                ks1, ks2),
                                requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(num_experts, out_channels),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x) #bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K,-1) #K,-1
        ks1, ks2 = self.kernel_size[0], self.kernel_size[1]
        aggregate_weight=torch.mm(softmax_att, weight).view(bs*self.out_planes,
                                                           self.in_planes//self.groups,
                                                           ks1,
                                                           ks2) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                            stride=self.stride, padding=self.padding,
                            groups=self.groups*bs,
                            dilation=self.dilation)
        else:
            output=F.conv2d(x, weight=aggregate_weight, bias=None, 
                            stride=self.stride, padding=self.padding, 
                            groups=self.groups*bs, 
                            dilation=self.dilation)
        
        n, c, h1, w1 = output.shape
        # output=output.view(bs, self.out_planes, h, w)
        output = output.view(bs, self.out_planes, h1, w1)
        return output