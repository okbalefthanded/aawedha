# implementation of EEG-Inception model in PyTorch
# adapted from the authors' Keras version:  https://github.com/esantamariavazquez/EEG-Inception/blob/main/EEGInception/EEGInception.py
# Santamaria-Vazquez, E., Martinez-Cagigal, V., Vaquerizo-Villar, F. and Hornero, R. (2020) 
# ‘EEG-Inception: A Novel Deep Convolutional Neural Network for Assistive ERP-Based Brain-Computer Interfaces’, 
# IEEE Transactions on Neural Systems and Rehabilitation Engineering, 28(12), pp. 2773–2782. 
# Available at: https://doi.org/10.1109/TNSRE.2020.3048106.
from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchdata import reshape_input
from aawedha.models.utils_models import is_a_loss
from torch import flatten
from torch import nn
import torch


def conv_block(in_channels, conv_type="Conv2D", filters=8, kernel=(1, 64), pad="same",
               bias=True, activation='elu', dropout_rate=0.2):
    if conv_type == "Conv2D":
        out = filters
        conv = nn.Conv2d(in_channels, out, kernel, padding=pad, bias=bias)
    else:
        # DepthWiseConv2D
        # out = in_channels * filters
        out = in_channels * 2
        conv = Conv2dWithConstraint(in_channels, out, kernel, padding=pad, 
                                    groups=in_channels, bias=False, max_norm=1)   
    
    return nn.Sequential(
                conv,
                nn.BatchNorm2d(out),
                nn.ELU(),
                nn.Dropout(p=dropout_rate)
                )

class EEGInceptionPTH(nn.Module):

    def __init__(self, nb_classes=1, Chans=15, Samples=205, name='EEGInceptionPTH'):
        super().__init__()
        self.name = name
        temp_f = 8
        spat_f = 2
        division_rate = 32
        kernel = 16
        # Block 1
        self.c1 = conv_block(1, "Conv2D", temp_f, (1, kernel*4), "same")
        self.d1 = conv_block(8,"Depth2D", spat_f, (Chans, 1), "valid")
        self.c2 = conv_block(1, "Conv2D", temp_f, (1, kernel*2), "same")
        self.d2 = conv_block(8,"Depth2D", spat_f, (Chans, 1), "valid")
        self.c3 = conv_block(1, "Conv2D", temp_f, (1, kernel), "same")
        self.d3 = conv_block(8,"Depth2D", spat_f, (Chans, 1), "valid")
        self.p1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        # Block 2
        self.c4 = conv_block(48, "Conv2D", temp_f, (1, kernel), "same")
        self.c5 = conv_block(48, "Conv2D", temp_f, (1, kernel // 2), "same")
        self.c6 = conv_block(48, "Conv2D", temp_f, (1, kernel // 4), "same")
        self.p2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Block 3 
        self.c7 = conv_block(24, "Conv2D", 12, (1, 8), "same", bias=False)
        self.p3 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.c8 = conv_block(12, "Conv2D", 6, (1, 4), "same", bias=False)
        self.p4 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dense = nn.Linear(6 * (Samples // division_rate), nb_classes)

        # initialize_Glorot_uniform(self)
        self.init_weights() # original paper inits Conv2D with He Normal

    def init_weights(self):
        for module in self.modules():
            if not is_a_loss(module):
                if hasattr(module, 'weight'):
                    # if not("BatchNorm" in module.__class__.__name__):
                    cls_name = module.__class__.__name__            
                    if not("BatchNorm" in cls_name or "LayerNorm" in cls_name):
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    else: 
                        nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)            

    def forward(self, x):
        x = reshape_input(x)
        #
        x1 = self.d1(self.c1(x))
        x2 = self.d2(self.c2(x))
        x3 = self.d3(self.c3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.p1(x)
        #
        x1 = self.c4(x)
        x2 = self.c5(x)
        x3 = self.c6(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.p2(x)
        #
        x = self.p3(self.c7(x))
        x = self.p4(self.c8(x))
        #
        x = flatten(x, 1)
        x = self.dense(x)
        return x