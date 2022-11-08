# Implements C-CNN model from:
# Ravi, A. et al. (2020) ‘Comparing user-dependent and user-independent 
# training of CNN for SSVEP BCI’, Journal of neural engineering. NLM (Medline), 
# 17(2), p. 026028. doi: 10.1088/1741-2552/ab6a67.
#
#
from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from torch import flatten
from torch import nn
import torch.nn.functional as F
import torch


class CCNN(TorchModel):

    def __init__(self, nb_classes=4, Chans=8, Samples=220, dropout_rate=0.25, kernLength=10,
                fs=512, resolution=0.293, device='cuda', name='CCNN'):
        super().__init__(device, name)
        filters = 2*Chans
        out_features = (Samples - (kernLength-1) - 1) + 1 
        self.fs = fs
        self.resolution = resolution
        self.nfft  = fs / resolution
        self.fft_start = 0 
        self.fft_end = 0
        self.conv1 = nn.Conv2d(Chans, filters, (Chans, 1), padding="valid")
        self.bn1   = nn.BatchNorm2d(filters)
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(filters, filters, (1, kernLength), padding="valid")
        self.bn2   = nn.BatchNorm2d(filters)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = LineardWithConstraint(filters * out_features, nb_classes)
        
        initialize_Glorot_uniform(self)

    def forward(self, x):
        x = self._reshape_input(x)
        x = self.transform(x)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = flatten(x, 1)
        x = self.fc(x)
        return x

    def transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, dim=-1) / samples
            real = x.real[]
            imag = x.image[]
            x = torch.cat((real, imag))
        return x
