from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.models.pytorch.prec_conv import PreConv
from antialiased_cnns import BlurPool
from torch.nn.functional import elu
from torch import nn
from torch import flatten
import torch.nn.functional as F
import torch

def MaxNorm(tensor, max_value):
    eps = 1e-7
    norms = torch.sqrt(torch.sum(torch.square(tensor), axis=0, keepdims=True))
    desired = torch.clip(norms, 0, max_value)
    return tensor * (desired / (norms + eps))    

# Adapted from braindecode
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = MaxNorm(self.weight.data, self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)       


class LineardWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LineardWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = MaxNorm(self.weight.data, self.max_norm)
        return super(LineardWithConstraint, self).forward(x)


class EEGNetTorchBase(TorchModel):

    def __init__(self, device="cuda", name="EEGNetTorch"):

        super().__init__(device=device, name=name)
        self.conv1 = None
        # self.bn1 = nn.BatchNorm2d(F1, momentum=0.99, eps=0.01) # same default values as TF
        # self.bn1 = nn.BatchNorm2d(F1)
        self.bn1 = None
        self.conv2 = None
        # self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.99, eps=0.01) # same default values as TF
        # self.bn2 = nn.BatchNorm2d(F1 * D)
        self.bn2 = None
        self.pool1 = None
        self.drop1 = None
        self.conv_sep_depth = None
        self.conv_sep_point = None
        # self.bn3 = nn.BatchNorm2d(F2, momentum=0.99, eps=0.01)
        # self.bn3 = nn.BatchNorm2d(F2)
        self.bn3 = None
        self.pool2 = None
        self.drop2 = None
        # self.dense = nn.Linear(nb_classes * (F2 * (Samples // 32)), nb_classes)
        # self.dense = LineardWithConstraint(nb_classes * (F2 * (Samples // 32)), nb_classes, max_norm=norm_rate)
        self.dense = None

        self.initialize_glorot_uniform()

    def forward(self, x):
        n, h, w = x.shape
        x = x.reshape(n, 1, h, w)
        x = self.bn1(self.conv1(x))
        x = elu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
    
        x = flatten(x, 1)
        x = self.dense(x)
        return x


class EEGNetTorch(EEGNetTorchBase):

    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25,
                 dropoutType='Dropout', device="cuda", name="EEGNetTorch"):

        super().__init__(device=device, name=name)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength),
                               bias=False, padding='same')
        # self.bn1 = nn.BatchNorm2d(F1, momentum=0.99, eps=0.01) # same default values as TF
        # self.bn1 = nn.BatchNorm2d(F1)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv2 = Conv2dWithConstraint(
            F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")
        # self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.99, eps=0.01) # same default values as TF
        # self.bn2 = nn.BatchNorm2d(F1 * D)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3) 
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, padding="same")
        self.conv_sep_point = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False, padding="valid")
        # self.bn3 = nn.BatchNorm2d(F2, momentum=0.99, eps=0.01)
        # self.bn3 = nn.BatchNorm2d(F2)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropoutRate)
        # self.dense = nn.Linear(nb_classes * (F2 * (Samples // 32)), nb_classes)
        # self.dense = LineardWithConstraint(nb_classes * (F2 * (Samples // 32)), nb_classes, max_norm=norm_rate)
        self.dense = LineardWithConstraint( (F2 * (Samples // 32)), nb_classes, max_norm=norm_rate)

        # self.initialize_glorot_uniform()


class EEGNetTorchSSVEP(EEGNetTorchBase):

    def __init__(self, nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, device="cuda", 
                 name="EEGNetTorchSSVEP"):

        super().__init__(device=device, name=name)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, padding="same")
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropoutRate)
        self.dense = nn.Linear((F2 * (Samples // 32)), nb_classes)


class EEGNetConvNorm(EEGNetTorchBase):

    def __init__(self, nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, affine=True, bn=True,
                 device="cuda", name="EEGNetTorchSSVEP"):

        super().__init__(device=device, name=name)

        self.conv1 = PreConv(1, F1, (1, kernLength), bias=False, padding='same',
                        affine=affine, bn=bn)
        self.conv2 = PreConv(F1, F1 * D, (Chans, 1), bias=False, groups=F1, padding="valid",
                             affine=affine, bn=bn)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = PreConv(F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, 
                                     padding="same", affine=False, bn=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropoutRate)
        self.dense = nn.Linear((F2 * (Samples // 32)), nb_classes)


class EEGNetTorchBlur(EEGNetTorchBase):

    def __init__(self, nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, device="cuda", 
                 name="EEGNetTorchBlurPool"):

        super().__init__(device=device, name=name)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = BlurPool(F1*D, stride=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, padding="same")
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, stride=(1, 8))
        self.drop2 = nn.Dropout(p=dropoutRate)
        self.dense = nn.Linear((F2 * (Samples // 32)), nb_classes)


class EEGNetBlurNorm(TorchModel):

    def __init__(self, nb_classes=12, Chans=8, Samples=256,
                 dropoutRate=0.5, kernLength=256, F1=96,
                 D=1, F2=96, affine=True, bn=True,
                 device="cuda", name="EEGNetBlurNorm"):

        super().__init__(device=device, name=name)

        self.conv1 = PreConv(1, F1, (1, kernLength), bias=False, padding='same',
                        affine=affine, bn=bn)
        self.conv2 = PreConv(F1, F1 * D, (Chans, 1), bias=False, groups=F1, padding="valid",
                             affine=affine, bn=bn)
        self.pool1 = BlurPool(F1*D, stride=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = PreConv(F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, 
                                     padding="same", affine=False, bn=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, stride=(1, 8))
        self.drop2 = nn.Dropout(p=dropoutRate)
        self.dense = nn.Linear((F2 * (Samples // 32)), nb_classes)
