from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from antialiased_cnns import BlurPool
from torch.nn.functional import elu
from torch import flatten
from torch import nn


class SghirNet(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=256, kernLength=256,
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

        self.initialize_glorot_uniform()
        

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

    def __init__(self, nb_classes, Chans=64, Samples=256, kernLength=256,
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
        self.pool1 = BlurPool(F2, kernel_size=(1, 2))
        self.do1   = nn.Dropout(p=dropoutRate)
        # block2
        self.conv2 = Conv2dWithConstraint(F2, F2, (1, kernLength // 4), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn2   = nn.BatchNorm2d(F2)
        self.pool2 = BlurPool(F2, kernel_size=(1, 2))
        self.do2   = nn.Dropout(p=dropoutRate)
        # block3
        self.conv3 = Conv2dWithConstraint(F2, F2, (1, kernLength // 16), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool3 = BlurPool(F2, kernel_size=(1, 2))
        self.do3   = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(F2, F2, (1, kernLength // 64), groups=D, bias=False, max_norm=1., padding="valid")
        self.bn4   = nn.BatchNorm2d(F2)
        self.pool4 = BlurPool(F2, kernel_size=(1, 2))
        self.do4   = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint((Samples // 8), nb_classes, max_norm=0.5)

        self.initialize_glorot_uniform()
        

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