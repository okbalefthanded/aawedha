# Implements LMDA-NET model from:
# Miao, Z., Zhang, X., Zhao, M. and Ming, D. (2023) ‘LMDA-Net:A lightweight multi-dimensional attention network for 
# general EEG-based brain-computer interface paradigms and interpretability’.
#  Available at: http://arxiv.org/abs/2303.16407.
# Adapted from official implementation: https://github.com/MiaoZhengQing/LMDA-Code/blob/main/lmda_model.py

import torch
import torch.nn as nn
from aawedha.trainers.torchdata import reshape_input
from aawedha.trainers.utils_models import is_a_loss

class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        # print('查看参数是否变化:', conv.bias)

        return y * self.C * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, nb_classes=4, 
                 Chans=8,  Samples=1125, 
                 depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5,
                return_features=False, 
                name="LMDA"):
        super(LMDA, self).__init__()
        self.name = name
        self.return_features = return_features
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, Chans), requires_grad=True)
        # nn.init.xavier_uniform_(self.channel_weight.data)
        # nn.init.kaiming_normal_(self.channel_weight.data, nonlinearity='relu')
        # nn.init.normal_(self.channel_weight.data)
        # nn.init.constant_(self.channel_weight.data, val=1/chans)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(Chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, Chans, Samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        # print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], nb_classes)

        self.init_weights()
            
    def init_weights(self):
        for module in self.modules():
            if not is_a_loss(module):
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.channel_weight.data)

    def forward(self, x):
        x = reshape_input(x)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        if self.return_features:
            return cls, features
        else:
            return cls