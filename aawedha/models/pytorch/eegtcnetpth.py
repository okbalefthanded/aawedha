from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from torch.nn.functional import elu
from torch import nn

# PyTorch Implementation  of EEGTCNet: 
# Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., &#38; Benini, L. (2020). 
# EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces. 
# IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2020-Octob, 2958–2965. 
# https://doi.org/10.1109/SMC42975.2020.9283028

# TCN Block and Convolution based on the original code : https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.elu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.elu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EEGTCNetPTH(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10, filt=10, 
             dropout=0, activation='relu', pooling='avg', F1=4, D=2, kernLength=64, 
             dropout_eeg=0.1, device="cuda", name="EEGTCNetPTH"):

        super().__init__(device=device, name=name)
        regRate = .25
        numFilters = F1
        F2 = numFilters*D
        # EEGNET
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")      
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop1 = nn.Dropout(p=dropout_eeg)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F2, F2, (1, 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout_eeg)
        # TCN
        self.tcn = TemporalConvNet(16, [filt]*layers, kernel_size=kernel_s, dropout=dropout)
        self.dense = LineardWithConstraint(filt, nb_classes, max_norm=regRate)
        #
        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)     
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))   
        x = elu(x)
        x = self.drop1(self.pool1(x))        
        x = self.conv_sep_point(self.conv_sep_depth(x))        
        x = self.bn3(x)
        x = elu(x)
        x = self.drop2(self.pool2(x))
        #
        n, c, _, w = x.shape
        x = x.reshape((n,c,w))
        #
        x = self.tcn(x)
        x = x[:,:,-1]
        x = self.dense(x)
        return x
    