# adapted from original paper code: https://github.com/xionghuiYu/EEG_CDILNet
# Implements: Liang T, Yu X, Liu X, Wang H, Liu X, Dong B. EEG-CDILNet: a lightweight and accurate 
# CNN network using circular dilated convolution for motor imagery classification. J Neural Eng. 2023;20(4). 
#

#padding = circular   can try padding  = reflective
import torch 
import torch.nn as nn 
from aawedha.trainers.torchdata import reshape_input
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class CDIL_Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dilation,  dropout=0.2,se = True):
        super(CDIL_Block, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride = stride,padding=padding, dilation=dilation, padding_mode='circular'))
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(self.conv1, self.batchnorm, self.relu1, self.dropout1)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()       

        # SE layers
        self.se = se
        if self.se:
            self.plane = n_outputs
            self.fc1 = nn.Conv1d(self.plane, self.plane//8, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
            self.fc2 = nn.Conv1d(self.plane//8, self.plane, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        """
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.se:
            # Squeeze
            w = F.avg_pool1d(out, out.size(2))
            w = F.relu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))

            # Excitation
            out = out * w  # New broadcasting feature from v0.2!

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) 

class CDIL_ConvPart(nn.Module): 
    def __init__(self, dim_in, hidden_channels, ks=3): 
        super(CDIL_ConvPart, self).__init__() 
        layers = [] 
        num_layer = len(hidden_channels) 
        for i in range(num_layer): 
            this_in = dim_in if i == 0 else hidden_channels[i - 1] 
            this_out = hidden_channels[i] 
            this_dilation = 2 ** i 
            this_padding = int(this_dilation * (ks - 1) / 2) 
            layers += [CDIL_Block(this_in, this_out, ks,1, this_padding, this_dilation)] 
        self.conv_net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.conv_net(x)


class EEG_CDILNet(nn.Module):
    
    def __init__(self, nb_classes=4, Chans=8, F1=24, KernLength = 64, D = 2, 
                 pe = 0.2, hiden=24, layer = 2, ks =3, pool = 8, name='EEG_CDILNET') -> None:
        super(EEG_CDILNet, self).__init__()
        self.name = name
        #
        F2 = F1 * D
        self.eegnet = nn.Sequential(
            #[b,1,c,t]  
            nn.ZeroPad2d((KernLength//2-1, KernLength//2, 0, 0)),
            nn.Conv2d(1, F1, (1, KernLength), bias=False, padding="same"),#   [b,24,c,t]
            nn.BatchNorm2d(F1), 
            #Depthwise Convolution
            nn.Conv2d(F1, F2, (Chans, 1), groups=F1, bias=False),#   [b,48,1,t]
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,pool)),     #   [b,48,1,t//8]
            nn.Dropout(pe),
            #Separable Convolution
            # nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F2, F2, (1, 16), groups=F2, bias=False, padding="same"),  #   [b,48,1,t//8]
            #Pointwise Convolution
            nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid"),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool)), #   [b,48,1,t//64]
            nn.Dropout(pe)
        )

        self.hiden = hiden
        self.layer = layer
        self.ks = ks
        
        self.conv = CDIL_ConvPart(F2, self.layer*[self.hiden], ks=self.ks)
        self.out = nn.Linear(self.hiden + F2, nb_classes)


    def forward(self, x):
        x = reshape_input(x)
        x = self.eegnet(x) #[b,48,1,t//64]

        in_cdil = torch.squeeze(x, dim=2)#[b,48,t//64]
        out_cdil = self.conv(in_cdil)
        con = torch.cat([in_cdil, out_cdil], dim=1)
        out = torch.mean(con, dim=2)    #[b,48,1]

        out = self.out(out)
        return out




