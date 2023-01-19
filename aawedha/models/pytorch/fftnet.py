from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torchdata import reshape_input
from torch import flatten
from torch import nn
import torch.nn.functional as F
import torch


class FFtNet(nn.Module):

    def __init__(self, nb_classes=12, Chans=8, kernLength=256, 
                 name="FFtNet"):

        super().__init__()
        self.name = name        
        self.length = kernLength
        self.conv1 = nn.Conv2d(1, 6, (Chans, 1), bias=False, padding='valid')
        self.bn1   = nn.BatchNorm2d(6)
        self.pool  = nn.MaxPool2d((1,3), stride=1)
        self.conv2 = nn.Conv2d(6, 16, (1, 70 // 2), padding='valid')
        self.bn2   = nn.BatchNorm2d(16)
        self.dense = nn.Linear(544, 15)
        self.drop  = nn.Dropout(0.5)
        self.dense2 = nn.Linear(15, nb_classes)
        
        initialize_Glorot_uniform(self)   

    def forward(self, x):        
        x = reshape_input(x)
        # 
        x = self.fft_transform(x)       
        #
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = flatten(x, 1)
        x = self.drop(self.dense(x))
        x = self.dense2(x)
        return x

    def fft_transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(torch.tensor(x), dim=-1) # LAST : samples
            x = x / samples
            x = 2*torch.abs(x)
            # f_input2 = f_input2[0:len(frequencies)]
            x = x[:,:,:, 0:70] # 70hz freq limit
        return x