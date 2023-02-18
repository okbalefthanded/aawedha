# Improving Complex features based CNN
#
from aawedha.models.pytorch.torchdata import reshape_input
from aawedha.models.utils_models import is_a_loss
from aawedha.layers.condconv import CondConv
from torchlayers.regularization import L2
from torch import flatten
from torch import nn
import torch.nn.functional as F
import torch


class SuperCCNN(nn.Module):

    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.25, kernLength=10,
                fs=512, resolution=0.293, l2=0.0001, frq_band=[7, 70], 
                name='SuperCCNN'):
        super().__init__()    
        self.name = name
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution)) 
        self.fft_end   = int(round(frq_band[1] / self.resolution)) + 1
         
        samples = (self.fft_end - self.fft_start) * 2        
        out_features = (samples - (kernLength-1) - 1) + 1 
        filters = 2*Chans

        self.conv1 = CondConv(1, filters, (Chans, 1), bias=False, groups=1, padding="valid", num_experts=4)
        self.ln1   = nn.LayerNorm([filters, 1, samples])
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = CondConv(filters, filters, (1, kernLength), bias=False, groups=1, padding="valid", num_experts=4)
        self.ln2   = nn.LayerNorm([filters, 1, out_features])
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc    = L2(nn.Linear(filters * out_features, nb_classes), weight_decay=l2)

        self.init_weights()
            
    def init_weights(self):
        for module in self.modules():
            if not is_a_loss(module):
                if hasattr(module, 'weight'):
                    cls_name = module.__class__.__name__            
                    if not("BatchNorm" in cls_name or "LayerNorm" in cls_name):
                        nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    else:
                        nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)    
    
    def forward(self, x):
        x = reshape_input(x)
        x = self.transform(x)
        x = self.drop1(F.relu(self.ln1(self.conv1(x))))
        x = self.drop2(F.relu(self.ln2(self.conv2(x))))    
        x = flatten(x, 1)
        x = self.fc(x)
        return x

    def transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, s=self.nfft, dim=-1) / samples
            real = x.real[:,:,:, self.fft_start:self.fft_end]
            imag = x.imag[:,:,:, self.fft_start:self.fft_end]
            x = torch.cat((real, imag), axis=-1)
        return x