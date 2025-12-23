"""
Implements: 
Dang, W., Li, M., Lv, D., Sun, X., &#38; Gao, Z. (2022). MHLCNN: Multi-Harmonic Linkage 
CNN Model for SSVEP and SSMVEP Signal Classification. 
IEEE Transactions on Circuits and Systems II: Express Briefs, 69(1), 244–248. 
https://doi.org/10.1109/TCSII.2021.3091803
"""
from aawedha.trainers.torchdata import reshape_input
from aawedha.trainers.utils_models import is_a_loss
from torch import flatten
from torch import nn
import numpy as np
import torch


class MHLCNN(nn.Module):

    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.5, m1=15, m2=20,
                 m3=25, fs=512, resolution=0.293, alpha1=[6, 15],
                 alpha2=[10, 27.5], alpha3=[15, 40], name='MHLCNN'):
        super().__init__()    
        self.name = name
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.alpha1 = [int(round(a / resolution)) for a in alpha1]
        self.alpha2 = [int(round(a / resolution)) for a in alpha2]
        self.alpha3 = [int(round(a / resolution)) for a in alpha3]
        
        N1 = 8
        N2 = 12
        N3 = 30
        samples1 = np.diff(self.alpha1).item()
        out_samples1 = samples1 - m1 + 1
        samples2 = np.diff(self.alpha2).item()
        out_samples2 = (samples2 - m2 + 1) // 2
        samples3 = np.diff(self.alpha3).item()
        out_samples3 = (samples3 - m3 + 1) // 3

        self.l1 = self.block(1, N1, N2, (Chans, 1) ,(1, m1), dropout_rate, (1,1))
        self.l2 = self.block(1, N1, N2, (Chans, 1) ,(1, m2), dropout_rate, (1,2))
        self.l3 = self.block(1, N1, N2, (Chans, 1) ,(1, m3), dropout_rate, (1,3))
        self.drop = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(N2 * (out_samples1 + out_samples2 + out_samples3), N3)
        self.fc2 = nn.Linear(N3, nb_classes)
        self.init_weights()
            
    def block(self, in1, out1, out2, k1, k2, drop_rate, pool_size):
      return nn.Sequential(
          nn.Conv2d(in1, out1, k1, padding="valid"),
          nn.ReLU(),
          nn.Dropout(drop_rate),
          nn.Conv2d(out1, out2, k2, padding="valid"),
          nn.ReLU(),
          nn.MaxPool2d(pool_size),
          nn.Dropout(drop_rate)
      )
    
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
        x1, x2, x3 = self.transform(x)
        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x3 = self.l3(x3)
        x1 = flatten(x1, 1)
        x2 = flatten(x2, 1)
        x3 = flatten(x3, 1)
        x = torch.cat((x1, x2, x3), axis=-1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, s=self.nfft, dim=-1) / samples
            x1 = x.real[:,:,:,self.alpha1[0]:self.alpha1[1]]
            x2 = x.real[:,:,:,self.alpha2[0]:self.alpha2[1]]
            x3 = x.real[:,:,:,self.alpha3[0]:self.alpha3[1]]
        return x1, x2, x3


"""
Improves: 
"""
class MHLCNN_plus(nn.Module):

    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.5, m1=15, m2=20,
                 m3=25, fs=512, resolution=0.293, alpha1=[6,15],
                 alpha2=[10, 27.5], alpha3=[15, 40], name='MHLCNN+'):
        super().__init__()    
        self.name = name
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.alpha1 = [int(round(a / resolution)) for a in alpha1]
        self.alpha2 = [int(round(a / resolution)) for a in alpha2]
        self.alpha3 = [int(round(a / resolution)) for a in alpha3]
        
        N1 = 8
        N2 = 12
        N3 = 30
        samples1 = np.diff(self.alpha1).item()*2
        out_samples1 = samples1 - m1 + 1
        samples2 = np.diff(self.alpha2).item()*2
        out_samples2 = (samples2 - m2 + 1) // 2
        samples3 = np.diff(self.alpha3).item()*2
        out_samples3 = (samples3 - m3 + 1) // 3

        self.l1 = self.block(1, N1, N2, (Chans, 1) ,(1, m1), dropout_rate, (1,1))
        self.l2 = self.block(1, N1, N2, (Chans, 1) ,(1, m2), dropout_rate, (1,2))
        self.l3 = self.block(1, N1, N2, (Chans, 1) ,(1, m3), dropout_rate, (1,3))
        self.drop = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(N2 * (out_samples1 + out_samples2 + out_samples3), N3)
        self.fc2 = nn.Linear(N3, nb_classes)
        
        self.init_weights()
            
    def block(self, in1, out1, out2, k1, k2, drop_rate, pool_size):
      return nn.Sequential(
          nn.Conv2d(in1, out1, k1, padding="valid", bias=False),
          nn.BatchNorm2d(out1),
          nn.ReLU(),
          nn.Dropout(drop_rate),
          nn.Conv2d(out1, out2, k2, padding="valid", bias=False),
          nn.BatchNorm2d(out2),
          nn.ReLU(),
          nn.MaxPool2d(pool_size),
          nn.Dropout(drop_rate)
      )
    
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
        x1, x2, x3 = self.transform(x)
        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x3 = self.l3(x3)
        x1 = flatten(x1, 1)
        x2 = flatten(x2, 1)
        x3 = flatten(x3, 1)
        x = torch.cat((x1, x2, x3), axis=-1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def transform(self, x):
        with torch.no_grad():
            samples = x.shape[-1]
            x = torch.fft.rfft2(x, s=self.nfft, dim=-1) / samples
            
            real = x.real[:,:,:, self.alpha1[0]:self.alpha1[1]]
            imag = x.imag[:,:,:, self.alpha1[0]:self.alpha1[1]]
            x1 = torch.cat((real, imag), axis=-1)

            real = x.real[:,:,:, self.alpha2[0]:self.alpha2[1]]
            imag = x.imag[:,:,:, self.alpha2[0]:self.alpha2[1]]
            x2 = torch.cat((real, imag), axis=-1)
            
            real = x.real[:,:,:, self.alpha3[0]:self.alpha3[1]]
            imag = x.imag[:,:,:, self.alpha3[0]:self.alpha3[1]]
            x3 = torch.cat((real, imag), axis=-1)
        return x1, x2, x3