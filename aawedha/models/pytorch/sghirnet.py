from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.layers.condconv import CondConv, CondConvConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.layers.cmt import Attention, Mlp
from timm.models.layers.drop import DropPath
from antialiased_cnns import BlurPool
from torch import flatten
from torch import nn
import torch.nn.functional as F
import torch


class SghirNet(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet"):
        super().__init__(device, name)
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform() 
        initialize_Glorot_uniform(self)      

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))        
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)        
        x = self.dense(x)
        return x


class SghirNet2(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))        
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)        
        x = self.dense(x)
        return x


class SghirNet3(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet3"):
        super().__init__(device, name)       
        # like a stem : temporal filters
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1 : spatial filters
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        # block2 : representations
        self.conv_sep_depth2 = nn.Conv2d(F2, F2, (1, kernLength // 4), bias=False, groups=F2, padding="same")
        self.conv_sep_point2 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        # block3
        self.conv_sep_depth3 = nn.Conv2d(F2, F2, (1, kernLength // 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point3 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv_sep_depth4 = nn.Conv2d(F2, F2, (1, kernLength // 64), bias=False, groups=F2, padding="same")
        self.conv_sep_point4 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint(F2*(Samples // 16), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()       
        initialize_Glorot_uniform(self)

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        
        x = self.conv_sep_point2(self.conv_sep_depth2(x))          
        x = self.do2(self.pool2(F.elu(self.bn2(x))))  

        x = self.conv_sep_point3(self.conv_sep_depth3(x)) 
        x = self.do3(self.pool3(F.elu(self.bn3(x))))
        
        x = self.conv_sep_point4(self.conv_sep_depth4(x))        
        x = self.do4(self.pool4(F.elu(self.bn4(x))))        
        
        x = flatten(x, 1)        
        x = self.dense(x)
        
        return x


class SghirNet4(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate)
        #
        self.dense = LineardWithConstraint((Samples // 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)

    def forward(self, x):
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = flatten(x, 1)
        x = self.dense(x)
        return x

def skip(in_plane, out_plane, length1, length2): 
    return nn.Sequential(
        nn.Conv2d(in_plane, out_plane, kernel_size=1, 
                  stride=length1 // length2, 
                  bias=False),
        nn.BatchNorm2d(out_plane)
    )


def skip2(in_plane, out_plane, length1, length2, dim): 
    return nn.Sequential(
        nn.Conv2d(in_plane, out_plane, kernel_size=1, 
                  stride=length1 // length2, 
                  bias=False),
        nn.LayerNorm(dim)
    )

# SghirNet + Skip connections
class SghirNet5(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet5"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 8, kernLength // 32)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # self.skip3 = skip()
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform() 
        initialize_Glorot_uniform(self)      

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)

        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)

        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        
        x = flatten(x, 1)        
        x = self.dense(x)
        return x


# SghirNet5 + SepatableDepthConv
class SghirNet6(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet5"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        # self.conv_sep_depth3 = nn.Conv2d(F2, F2, (1, kernLength // 16), bias=False, groups=F2, padding="same")
        # self.conv_sep_point3 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # self.skip3 = skip()
        # block4
        self.conv_sep_depth4 = nn.Conv2d(F2, F2, (1, kernLength // 64), bias=False, groups=F2, padding="same")
        self.conv_sep_point4 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint(F2*(Samples // 64), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform() 
        initialize_Glorot_uniform(self)      

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))  
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)        
        # x = self.conv_sep_point3(self.conv_sep_depth3(x)) 
        # x = self.do3(self.pool3(elu(self.bn3(x))))
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))         
        x = x + self.skip2(shortcut2)      
        x = self.conv_sep_point4(self.conv_sep_depth4(x))
        x = self.do4(self.pool4(F.elu(self.bn4(x))))                
        x = flatten(x, 1)        
        x = self.dense(x)
        return x

# sghirnet5 + BlurPool
class SghirNet7(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# SghirNet 7 + 3rd skip
class SghirNet8(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate)
        self.skip3 = skip(F2, F2, kernLength // 2, kernLength // 8) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))        
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)        
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = x + self.skip2(shortcut2)        
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))
        x = x + self.skip3(shortcut3)    
        x = flatten(x, 1)
        x = self.dense(x)
        return x

# sghirnet7 with BN -> LN
class SghirNet9(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        # self.bn1   = nn.BatchNorm2d(F2)
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        # self.bn2   = nn.BatchNorm2d(F2)
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        # self.bn3   = nn.BatchNorm2d(F2)
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        # self.bn4   = nn.BatchNorm2d(F2)
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet7 as ConvNext
class SghirNet10(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet10"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 16)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 64)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

class SghirNet10dw(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet10dw"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=F2, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 16)+1), groups=F2, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 64)+1), groups=F2, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 4), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet7 : Conv2d -> CondConv
class SghirNet11(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1  
        # cc = CondConv(1, 32, (1, 256), bias=False, padding='valid', num_experts=4)      
        self.conv1 = CondConv(F1, F2, (Chans, 1), bias=False, groups=D, padding="valid", num_experts=4)
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = CondConv(F2, F2, (1, kernLength // 4), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = CondConv(F2, F2, (1, kernLength // 16), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = CondConv(F2, F2, (1, kernLength // 64), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet11 + 10
class SghirNet12(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1  
        # cc = CondConv(1, 32, (1, 256), bias=False, padding='valid', num_experts=4)      
        self.conv1 = CondConv(F1, F2, (Chans, 1), bias=False, groups=D, padding="valid", num_experts=4)
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = CondConv(F2, F2, (1, kernLength // 4), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = CondConv(F2, F2, (1, kernLength // 16), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = CondConv(F2, F2, (1, kernLength // 64), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet11 with Constrained ConvConv
class SghirNet13(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet13"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1  
        # cc = CondConv(1, 32, (1, 256), bias=False, padding='valid', num_experts=4)      
        self.conv1 = CondConvConstraint(F1, F2, (Chans, 1), bias=False, groups=D, padding="valid", num_experts=4)
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = CondConvConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = CondConvConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = CondConvConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet11 with Constrained ConvConv and as ConvNext
class SghirNet14(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet14"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1  
        # cc = CondConv(1, 32, (1, 256), bias=False, padding='valid', num_experts=4)      
        self.conv1 = CondConvConstraint(F1, F2, (Chans, 1), bias=False, groups=D, padding="valid", num_experts=4)
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = CondConvConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength//4) + 1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = CondConvConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength//16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = CondConvConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength//64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        
        x = self.dense(x)
        return x

class SghirNet14_2(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet14_2"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1  
        # cc = CondConv(1, 32, (1, 256), bias=False, padding='valid', num_experts=4)      
        self.conv1 = CondConvConstraint(F1, F2, (Chans, 1), bias=False, groups=D, padding="valid", num_experts=4)
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = CondConvConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength//4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = CondConvConstraint(F2, F2, (1, (kernLength // 16)+1), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength//16)])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = CondConvConstraint(F2, F2, (1, (kernLength // 64)+1), groups=D, bias=False, padding="valid", num_experts=4)
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength//64)])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        
        x = self.dense(x)
        return x

# sghirnet10 with last Convs as sepconv
class SghirNet15(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv_sep_depth4 = nn.Conv2d(F2, F2, (1, kernLength // 64), bias=False, groups=F2, padding="same")
        self.conv_sep_point4 = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 32)])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint(F2*(Samples // 64), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))

        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.conv_sep_point4(self.conv_sep_depth4(x))
        x = self.do4(self.pool4(F.sgelu(self.bn4(x))))
                
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# SghirNet10 + skip has LN 
class SghirNet16(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet2"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip2(F2, F2, kernLength // 2, kernLength // 8, [F2, 1, kernLength //8])
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip2(F2, F2, kernLength // 2, kernLength // 8, [F2, 1, kernLength //32])
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# SghirNet10 + DropPath in skips
class SghirNet17(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, dropPath_rate=0.2,
                device="cuda", name="SghirNet17"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.dp1   = DropPath(dropPath_rate) if dropPath_rate else nn.Identity()
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.dp2   = DropPath(dropPath_rate) if dropPath_rate else nn.Identity()
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.dp1(x) + self.skip1(shortcut1)
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = self.dp2(x) + self.skip2(shortcut2)
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# SghirNet 10 + 3rd skip + drop path
class SghirNet18(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, dropPath_rate=0.2, 
                device="cuda", name="SghirNet18"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.dp1   = DropPath(dropPath_rate) if dropPath_rate else nn.Identity()
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.dp2   = DropPath(dropPath_rate) if dropPath_rate else nn.Identity()
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        self.skip3 = skip(F2, F2, kernLength // 2, kernLength // 8) 
        self.dp3   = DropPath(dropPath_rate) if dropPath_rate else nn.Identity()
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))        
        
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.dp1(x) + self.skip1(shortcut1)        
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = self.dp2(x) + self.skip2(shortcut2)        
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))
        x = self.dp3(x) + self.skip3(shortcut3)    
        
        x = flatten(x, 1)
        x = self.dense(x)
        return x

# MultiBranch SghirNet10
def conv_block(in_channels, filters=8, kernel=(1, 64), pad="valid",
               temp=1024, activation=nn.GELU, dropout_rate=0.5):    
    return nn.Sequential(
                Conv2dWithConstraint(in_channels, filters, kernel, padding=pad, 
                                    groups=1, bias=False, max_norm=1),
                nn.LayerNorm([filters, 1, temp]),
                activation(),
                nn.Dropout(p=dropout_rate)
                )
    
def stem_block(in_channels, filters=8, kernel=(1, 64)):    
    return nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel, padding="same", bias=False),
                nn.BatchNorm2d(filters)
                )

class SghirNet19(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet19"):
        super().__init__(device, name)    
        expansion = F2*3
        # like a stem
        self.conv01 = stem_block(1, F1, (1, kernLength // 4))
        self.conv02 = stem_block(1, F1, (1, kernLength // 2))
        self.conv03 = stem_block(1, F1, (1, kernLength))
        # block1  
        self.conv11 = conv_block(F1*3, F2, (Chans, 1), dropout_rate=dropoutRate) 
        self.conv12 = conv_block(F1*3, F2, (Chans, 1), dropout_rate=dropoutRate) 
        self.conv13 = conv_block(F1*3, F2, (Chans, 1), dropout_rate=dropoutRate)    
        self.pool1  = BlurPool(expansion, filt_size=(1,2), stride=(1,2))
        self.skip1  = skip(expansion, expansion, kernLength // 2, kernLength // 4)
        # block2
        self.conv21 = conv_block(expansion, F2, (1, kernLength // 2), temp=kernLength // 2, pad='same', dropout_rate=dropoutRate)
        self.conv22 = conv_block(expansion, F2, (1, kernLength // 4), temp=kernLength // 2, pad='same', dropout_rate=dropoutRate)
        self.conv23 = conv_block(expansion, F2, (1, kernLength // 8), temp=kernLength // 2, pad='same', dropout_rate=dropoutRate)
        self.pool2  = BlurPool(expansion, filt_size=(1,2), stride=(1,2))
        self.skip2  = skip(expansion, expansion, kernLength // 2, kernLength // 4)      
        # block3
        self.conv31 = conv_block(expansion, F2, (1, kernLength // 8), temp=kernLength // 4, pad='same', dropout_rate=dropoutRate)
        self.conv32 = conv_block(expansion, F2, (1, kernLength // 4), temp=kernLength // 4, pad='same', dropout_rate=dropoutRate)
        self.conv33 = conv_block(expansion, F2, (1, kernLength // 16), temp=kernLength // 4, pad='same', dropout_rate=dropoutRate)
        self.pool3  = BlurPool(expansion, filt_size=(1,2), stride=(1,2))
        # block4
        self.conv41 = conv_block(expansion, F2, (1, kernLength // 16), temp=kernLength // 8, pad='same', dropout_rate=dropoutRate)
        self.conv42 = conv_block(expansion, F2, (1, kernLength // 32), temp=kernLength // 8, pad='same', dropout_rate=dropoutRate)
        self.conv43 = conv_block(expansion, F2, (1, kernLength // 64), temp=kernLength // 8, pad='same', dropout_rate=dropoutRate)
        self.pool4  = BlurPool(expansion, filt_size=(1,2), stride=(1,2))
        #
        self.dense = LineardWithConstraint(expansion*2*(Samples // 16), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        
        x1 = self.conv01(x)
        x2 = self.conv02(x)
        x3 = self.conv03(x)
        x =  torch.cat((x1, x2, x3), dim=1)
        
        x1 = self.conv11(x)
        x2 = self.conv12(x)
        x3 = self.conv13(x)
        x =  torch.cat((x1, x2, x3), dim=1)
        x = self.pool1(x)
        shortcut1 = x
        
        x1 = self.conv21(x)
        x2 = self.conv22(x)
        x3 = self.conv23(x)
        x =  torch.cat((x1, x2, x3), dim=1)
        x = self.pool2(x)
        shortcut2 = x
        x = x + self.skip1(shortcut1)

        x1 = self.conv31(x)
        x2 = self.conv32(x)
        x3 = self.conv33(x)
        x =  torch.cat((x1, x2, x3), dim=1)
        x = self.pool3(x)
        x = x + self.skip2(shortcut2)

        x1 = self.conv41(x)
        x2 = self.conv42(x)
        x3 = self.conv43(x)
        x =  torch.cat((x1, x2, x3), dim=1)        
        
        x = flatten(x, 1)       
        x = self.dense(x)
        
        return x

# SghirNet10 + LN between Blocks, 
# imitating Vision transofrmers
class SghirNet20(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet20"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln1   = nn.LayerNorm([F2, 1, kernLength //8]) # nn.LayerNorm([F2, 1, kernLength])
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)        
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln2   = nn.LayerNorm([F2, 1, (kernLength // 32)])
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        # self.ln3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate)
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        # self.ln4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate)
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))

        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.ln1(x + self.skip1(shortcut1))
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = self.ln2(x + self.skip2(shortcut2))
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x


# SghirNet 10 + 3rd skip + LN 
class SghirNet21(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, dropPath_rate=0.2, 
                device="cuda", name="SghirNet21"):
        super().__init__(device, name)       
        # like a stem
        # self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=1.)
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.BatchNorm2d(F2)
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln1   = nn.LayerNorm([F2, 1, kernLength //8])
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln2   = nn.LayerNorm([F2, 1, (kernLength // 32)])
        
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        self.skip3 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln3   = nn.LayerNorm([F2, 1, (kernLength // 128)])
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        # self.ln4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))        
        
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.ln1(x + self.skip1(shortcut1))        
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = self.ln2(x + self.skip2(shortcut2))   
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))
        x = self.ln3(x + self.skip3(shortcut3))    
        
        x = flatten(x, 1)
        x = self.dense(x)
        return x


# SghirNet20 + LN everywhere
class SghirNet22(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet22"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln1   = nn.LayerNorm([F2, 1, kernLength //8]) # 
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])    
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln2   = nn.LayerNorm([F2, 1, (kernLength // 32)])
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate)
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate)
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))

        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.ln1(x + self.skip1(shortcut1))
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = self.ln2(x + self.skip2(shortcut2))
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x


# SghirNet 21 LN  everywhere
class SghirNet23(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, dropPath_rate=0.2, 
                device="cuda", name="SghirNet23"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln1   = nn.LayerNorm([F2, 1, kernLength //8])
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)+1])   
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln2   = nn.LayerNorm([F2, 1, (kernLength // 32)])
        
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)+1])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        self.skip3 = skip(F2, F2, kernLength // 2, kernLength // 8)
        self.ln3   = nn.LayerNorm([F2, 1, (kernLength // 128)])
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   =  nn.LayerNorm([F2, 1, (kernLength // 64)+1])
        self.pool4 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))        
        
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.ln1(x + self.skip1(shortcut1))        
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = self.ln2(x + self.skip2(shortcut2))   
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))
        x = self.ln3(x + self.skip3(shortcut3))    
        
        x = flatten(x, 1)
        x = self.dense(x)
        return x

# sghirnet10 incremental filters
class SghirNet24(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet24"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2*2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2*2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2*2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2*2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2*2, F2*3, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2*2, F2*3, (1, (kernLength // 16)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2*3, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2*3, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2*3, F2*4, (1, (kernLength // 64)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2*4, 1, (kernLength // 64)])
        self.pool4 = BlurPool(F2*4, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        
        x = flatten(x, 1)  
        x = self.dense(x)
        return x

# sghirnet10 decremental filters
class SghirNet25(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, device="cuda", 
                name="SghirNet25"):
        super().__init__(device, name)       
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2//2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2//2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2//2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2//2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2//2, F2//4, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2//2, F2//4, (1, (kernLength // 16)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2//4, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2//4, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2//4, F2//8, (1, (kernLength // 64)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.LayerNorm([F2//8, 1, (kernLength // 64)])
        self.pool4 = BlurPool(F2//8, filt_size=(1,2), stride=(1,2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 64), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))        
        
        x = flatten(x, 1)  
        x = self.dense(x)
        return x

# SghirNet10 + Attention @ last layer
class SghirNet26(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, 
                heads=1, dim=46, 
                device="cuda", name="SghirNet26"):
        super().__init__(device, name)       
        pos_shape = kernLength // 32
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 16)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4        
        self.ln1   = nn.LayerNorm(F2)
        self.relative_pos = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        #
        self.dense = LineardWithConstraint((Samples // 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        # initialize_He_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)   
        
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        
        x = self.attn(self.ln1(x), H, W, self.relative_pos)     
        x = flatten(x, 1)        

        x = self.dense(x)
        return x

# SghirNet10 + 2 Attention Layers
class SghirNet27(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, 
                heads=1, dim=46, 
                device="cuda", name="SghirNet27"):
        super().__init__(device, name)       
        pos_shape1 = kernLength // 8
        # pos_shape2 = kernLength // 32        
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.ln1   = nn.LayerNorm(F2)
        self.relative_pos1 = nn.Parameter(torch.randn(heads, pos_shape1, pos_shape1)) 
        self.attn1   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)

        # block4        
        self.ln2   = nn.LayerNorm(F2)
        self.relative_pos2 = nn.Parameter(torch.randn(heads, pos_shape1, pos_shape1)) 
        self.attn2   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        #
        self.dense = LineardWithConstraint((Samples * 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        # initialize_He_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        x = x + self.skip1(shortcut1)
        
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        
        x = self.attn1(self.ln1(x), H, W, self.relative_pos1)
        
        x = self.attn2(self.ln2(x), H, W, self.relative_pos2)     
        
        x = flatten(x, 1)       

        x = self.dense(x)
        return x

# SghirNet26 + MLP after Attention
class SghirNet28(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, 
                heads=1, dim=46, 
                device="cuda", name="SghirNet26"):
        super().__init__(device, name)       
        pos_shape = kernLength // 32
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 16)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.LayerNorm([F2, 1, (kernLength // 16)])
        self.pool3 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4        
        self.ln1   = nn.LayerNorm(F2)
        self.relative_pos = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn  = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        self.ln2   = nn.LayerNorm(F2)
        self.mlp   = Mlp(F2) 
        #

        #
        self.dense = LineardWithConstraint((Samples // 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        # initialize_He_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(F.gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        # x = self.do4(self.pool4(F.gelu(self.bn4(self.conv4(x)))))   
        
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        
        x = self.attn(self.ln1(x), H, W, self.relative_pos)  
        x = self.mlp(self.ln2(x), H, W)

        x = flatten(x, 1)      
        x = self.dense(x)
        return x


# SghirNet 27 + MLP after Attention
class SghirNet29(TorchModel):

    def __init__(self, nb_classes=4, Chans=64, Samples=256, kernLength=256,
                F1=32, F2=16, D=1, dropoutRate=0.5, 
                heads=1, dim=46, device="cuda", name="SghirNet29"):
        super().__init__(device, name)       
        pos_shape1 = kernLength // 8
        # pos_shape2 = kernLength // 32        
        # like a stem
        self.conv = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn   = nn.BatchNorm2d(F1)
        # block1        
        self.conv1 = Conv2dWithConstraint(F1, F2, (Chans, 1), max_norm=1, bias=False, groups=D, padding="valid")
        self.bn1   = nn.LayerNorm([F2, 1, kernLength])
        self.pool1 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do1   = nn.Dropout(p=dropoutRate)
        self.skip1 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, (kernLength // 4)+1), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.LayerNorm([F2, 1, (kernLength // 4)])
        self.pool2 = BlurPool(F2, filt_size=(1,2), stride=(1,2))
        self.do2   = nn.Dropout(p=dropoutRate)
        self.skip2 = skip(F2, F2, kernLength // 2, kernLength // 8)
        # block3
        self.ln1   = nn.LayerNorm(F2)
        self.relative_pos1 = nn.Parameter(torch.randn(heads, pos_shape1, pos_shape1)) 
        self.attn1   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        self.ln2   = nn.LayerNorm(F2)
        self.mlp   = Mlp(F2) 

        # block4        
        self.ln3   = nn.LayerNorm(F2)
        self.relative_pos2 = nn.Parameter(torch.randn(heads, pos_shape1, pos_shape1)) 
        self.attn2   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        self.ln4   = nn.LayerNorm(F2)
        self.mlp2  = Mlp(F2) 
        #
        self.dense = LineardWithConstraint((Samples * 2), nb_classes, max_norm=0.5)

        # self.initialize_glorot_uniform()
        initialize_Glorot_uniform(self)
        # initialize_He_uniform(self)
        
    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.bn(self.conv(x))
                
        x = self.do1(self.pool1(F.gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(F.gelu(self.bn2(self.conv2(x)))))        
        x = x + self.skip1(shortcut1)
        
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        
        x = self.attn1(self.ln1(x), H, W, self.relative_pos1)
        x = self.mlp(self.ln2(x), H, W)
        
        x = self.attn2(self.ln3(x), H, W, self.relative_pos2) 
        x = self.mlp2(self.ln2(x), H, W)    
        
        x = flatten(x, 1)       

        x = self.dense(x)
        return x


# Complex-features on CMT
class SghirNet30(TorchModel):

    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.25,
                 fs=512, resolution=0.293,frq_band=[7, 70], heads=1,
                 device='cuda', name='SghirNet30'):
        super().__init__(device, name)

        self.fs = fs
        self.resolution = resolution
        self.nfft       = round(fs / resolution)
        self.fft_start  = int(round(frq_band[0] / self.resolution)) 
        self.fft_end    = int(round(frq_band[1] / self.resolution)) + 1
        
        samples = (self.fft_end - self.fft_start) * 2        
        filters = 2*Chans

        self.conv1 = Conv2dWithConstraint(1, filters, (Chans, 1), max_norm=1, bias=False, padding="valid")
        self.bn1   = nn.LayerNorm([filters, 1, samples])
        self.do1   = nn.Dropout(p=dropout_rate)
        
        self.ln1   = nn.LayerNorm(filters)
        self.relative_pos1 = nn.Parameter(torch.randn(heads, samples, samples)) 
        self.attn1   = Attention(dim=filters, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)

        self.ln2   = nn.LayerNorm(filters)
        self.relative_pos2 = nn.Parameter(torch.randn(heads, samples, samples)) 
        self.attn2   = Attention(dim=filters, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        
        self.dense = LineardWithConstraint(filters * samples, nb_classes, max_norm=0.5)

        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)
        x = self.transform(x)
        x = self.do1(self.bn1(self.conv1(x)))

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        shortcut1 = x
        x = self.attn1(self.ln1(x), H, W, self.relative_pos1)
        x = x + shortcut1
        shortcut2 = x
        x = self.attn2(self.ln2(x), H, W, self.relative_pos2)
        x = x + shortcut2

        x = flatten(x, 1)
        x = self.dense(x)       
        
        return x 

    def transform(self, x):
      with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, s=self.nfft, dim=-1) / samples
            real = x.real[:,:,:, self.fft_start:self.fft_end]
            imag = x.imag[:,:,:, self.fft_start:self.fft_end]
            x = torch.cat((real, imag), axis=-1)
      return x