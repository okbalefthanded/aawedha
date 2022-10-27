from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.layers.condconv import CondConv, CondConvConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from timm.models.layers.drop import DropPath
from torch.nn.functional import elu, gelu
from antialiased_cnns import BlurPool
from torch import flatten
from torch import nn


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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))        
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))        
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
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
        
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))
        
        x = self.conv_sep_point2(self.conv_sep_depth2(x))          
        x = self.do2(self.pool2(elu(self.bn2(x))))  

        x = self.conv_sep_point3(self.conv_sep_depth3(x)) 
        x = self.do3(self.pool3(elu(self.bn3(x))))
        
        x = self.conv_sep_point4(self.conv_sep_depth4(x))        
        x = self.do4(self.pool4(elu(self.bn4(x))))        
        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
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
        
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)

        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)

        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)        
        # x = self.conv_sep_point3(self.conv_sep_depth3(x)) 
        # x = self.do3(self.pool3(elu(self.bn3(x))))
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))         
        x = x + self.skip2(shortcut2)      
        x = self.conv_sep_point4(self.conv_sep_depth4(x))
        x = self.do4(self.pool4(elu(self.bn4(x))))                
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)        
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = x + self.skip2(shortcut2)        
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
        x = flatten(x, 1)       
        x = self.dense(x)
        return x

# sghirnet7 as ConvNext
class SghirNet10(TorchModel):

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
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))        
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
        
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))        
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

        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        
        x = self.conv_sep_point4(self.conv_sep_depth4(x))
        x = self.do4(self.pool4(gelu(self.bn4(x))))
                
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
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = x + self.skip1(shortcut1)
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = x + self.skip2(shortcut2)
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))        
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
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.dp1(x) + self.skip1(shortcut1)
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        x = self.dp2(x) + self.skip2(shortcut2)
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))        
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
        
        x = self.do1(self.pool1(gelu(self.bn1(self.conv1(x)))))        
        shortcut1 = x        
        
        x = self.do2(self.pool2(gelu(self.bn2(self.conv2(x)))))        
        shortcut2 = x
        x = self.dp1(x) + self.skip1(shortcut1)        
        
        x = self.do3(self.pool3(gelu(self.bn3(self.conv3(x)))))
        shortcut3 = x
        x = self.dp2(x) + self.skip2(shortcut2)        
        
        x = self.do4(self.pool4(gelu(self.bn4(self.conv4(x)))))
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