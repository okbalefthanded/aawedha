# Implement model FBCCNN of : https://www.mdpi.com/2076-3425/13/5/780

from aawedha.trainers.torchdata import reshape_input
from aawedha.trainers.utils_models import is_a_loss
from scipy.signal import butter, filtfilt
from torchlayers.regularization import L2
from torch import flatten
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


# TODO: Implement the class FBCCNN
class FBCCNN(nn.Module):

    def __init__(self, 
                 nb_classes=4, 
                 Chans=8,  
                 dropout_rate=0.25, 
                 kernLength=10,
                 fs=512, 
                 resolution=0.293, 
                 l2=0.0001, 
                 frq_band=[7, 70], 
                 subbands=3,
                 return_features=False, 
                name='FBCCNN'):
        super().__init__()    
        self.name = name
        self.return_features = return_features
        self.fs = fs
        self.band = [ [frq_band[0]*i, frq_band[1]] for i in range(1, subbands+1)] 
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution))  
        self.fft_end = int(round(frq_band[1] / self.resolution)) + 1
        
        samples = (self.fft_end - self.fft_start) * 2        
        filters = 64 # 2*Chans
        k2 = 64
        s2 = (64, 64)
        out_features = (((samples*subbands - k2) / s2[0]) + 1) * k2 
        self.conv1 = nn.Conv2d(1, filters, (Chans, 1), bias=False, padding="valid")
        self.bn1   = nn.BatchNorm2d(filters)
        self.act1   = nn.PReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(filters, filters, (1, k2), bias=False, stride=s2, padding="valid")
        self.bn2   = nn.BatchNorm2d(filters)
        self.act2  = nn.PReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = nn.Linear(out_features, nb_classes)                        

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
        out = []
        for b in self.band:
            c = self.filter_band(x, b)
            c = self.transform(c)
            out.append(c)
        x = torch.cat(out, axis=-1)
        x = self.drop1(self.act1(self.bn1(self.conv1(x))))
        x = self.drop2(self.act2(self.bn2(self.conv2(x))))
        x = flatten(x, 1)
        features = x
        x = self.fc(x)
        if self.return_features:
            return x, features
        else:
            return x

    def transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, s=self.nfft, dim=-1) / samples
            real = x.real[:,:,:, self.fft_start:self.fft_end]
            imag = x.imag[:,:,:, self.fft_start:self.fft_end]
            x = torch.cat((real, imag), axis=-1)
        return x
    
    def filter_band(self, x, band):
        # x: batch, channels, samples
        device = x.device
        with torch.no_grad():
            x = x.cpu().numpy()
            B, A = butter(4, np.array(band) / (self.fs / 2), btype='bandpass')
            x = filtfilt(B, A, x, axis=-1)
            x = x.copy()
        return torch.tensor(x, dtype=torch.float, device=device)