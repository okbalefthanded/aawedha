# Implements C-CNN model from:
# Ravi, A. et al. (2020) ‘Comparing user-dependent and user-independent 
# training of CNN for SSVEP BCI’, Journal of neural engineering. NLM (Medline), 
# 17(2), p. 026028. doi: 10.1088/1741-2552/ab6a67.
#
#
from aawedha.trainers.torchdata import reshape_input
from aawedha.trainers.utils_models import is_a_loss
from scipy.signal import butter, filtfilt, welch
from torchlayers.regularization import L2
from torch import flatten
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


def transform(x, nfft, start, end):
    xt = torch.tensor(x)
    with torch.no_grad():
        samples = x.shape[0]
        xt = torch.fft.rfft2(xt, s=nfft, dim=0) / samples
        real = xt.real[start:end, :, :]
        imag = xt.imag[start:end, :, :]
        xt = torch.cat((real, imag), axis=0)
    return xt.detach().numpy()

class CCNN(nn.Module):

    def __init__(self, 
                 nb_classes=4, 
                 Chans=8,  
                 dropout_rate=0.25, 
                 kernLength=10,
                 fs=512, 
                 resolution=0.293, 
                 l2=0.0001, 
                 frq_band=[7, 70], 
                 return_features=False, 
                name='CCNN'):
        super().__init__()    
        self.name = name
        self.return_features = return_features
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution)) 
        self.fft_end = int(round(frq_band[1] / self.resolution)) + 1
        
        samples = (self.fft_end - self.fft_start) * 2        
        # out_features = samples - kernLength - 1  
        out_features = (samples - (kernLength-1) - 1) + 1 
        filters = 2*Chans

        self.conv1 = L2(nn.Conv2d(1, filters, (Chans, 1), bias=False, padding="valid"), 
                        weight_decay=l2)
        self.bn1   = nn.BatchNorm2d(filters)
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = L2(nn.Conv2d(filters, filters, (1, kernLength), bias=False, padding="valid"),
                        weight_decay=l2)
        self.bn2   = nn.BatchNorm2d(filters)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = L2(nn.Linear(filters * out_features, nb_classes), 
                        weight_decay=l2)

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
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
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

class CCNNw(nn.Module):

    def __init__(self, 
                 nb_classes=4, 
                 Chans=8,  
                 dropout_rate=0.25, 
                 kernLength=10,
                 fs=512, 
                 resolution=0.293, 
                 l2=0.0001, 
                 frq_band=[7, 70], 
                 return_features=False, 
                name='CCNN'):
        super().__init__()    
        self.name = name
        self.return_features = return_features
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution)) 
        self.fft_end = int(round(frq_band[1] / self.resolution)) + 1
        
        samples = (self.fft_end - self.fft_start) * 2        
        # out_features = samples - kernLength - 1  
        out_features = (samples - (kernLength-1) - 1) + 1 
        filters = 2*Chans

        self.conv1 = L2(nn.Conv2d(1, filters, (Chans, 1), bias=False, padding="valid"), 
                        weight_decay=l2)
        self.bn1   = nn.BatchNorm2d(filters)
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = L2(nn.Conv2d(filters, filters, (1, kernLength), bias=False, padding="valid"),
                        weight_decay=l2)
        self.bn2   = nn.BatchNorm2d(filters)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = L2(nn.Linear(filters * out_features, nb_classes), 
                        weight_decay=l2)

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
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = flatten(x, 1)
        features = x
        x = self.fc(x)
        if self.return_features:
            return x, features
        else:
            return x
        
    def transform(self, x):
        device = x.device
        with torch.no_grad():            
            samples = x.shape[-1]       
            _, x = welch(x, fs = self.fs, nperseg = samples, window = 'hann', noverlap = None, 
                        nfft = self.nfft,
                        scaling = 'spectrum', 
                        return_onesided=True, 
                        axis=-1)#
            x = x.cpu().numpy()
            y = torch.fft.rfft2(x.to(device), s=self.nfft, dim=-1) / samples
            real = torch.tensor(x.copy(), dtype=torch.float, device=device)
            real = real[:,:,:, self.fft_start:self.fft_end]
            imag = y.imag[:,:,:, self.fft_start:self.fft_end]             
            x = torch.cat((real, imag), axis=-1)
        return x
    
class FBLCCNN(nn.Module):
  def __init__(self, fs=256, nb_classes=12, n_subbands=3, 
               band=[8, 80], models=None, name="FB-LCCNN"):
    super().__init__()
    self.name = name
    self.fs = fs
    self.subbands = [[band[0]*i, band[1]] for i in range(1, n_subbands+1)]
    self.subnets  = models
    self.conv     = nn.Conv1d(n_subbands, 1, 1, padding='same')
    self.in_features = None
    self.out_features = nb_classes # number of classes
    self.init_weights()

  def init_weights(self):
    nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
    nn.init.constant_(self.conv.bias, 0)

  def freeze_models(self):
    for model in self.subnets:
      for param in model.parameters():
        param.requires_grad = False
  
  def forward(self, x):
    out = []
    for i, band in enumerate(self.subbands):
      c = self.filter_band(x, band)
      c = self.subnets[i](c)
      c = c.unsqueeze(1)
      out.append(c)
    #
    x = torch.cat(out, 1)
    x = self.conv(x)
    return x.squeeze(1)

  def filter_band(self, x, band):
    # x: batch, channels, samples
    device = x.device
    with torch.no_grad():
      x = x.cpu().numpy()
      B, A = butter(4, np.array(band) / (self.fs / 2), btype='bandpass')
      x = filtfilt(B, A, x, axis=-1)
      x = x.copy()
    return torch.tensor(x, dtype=torch.float, device=device)


class CCNN_sghir(nn.Module):
    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.25, kernLength=10,
                fs=512, resolution=0.293, l2=0.0001, frq_band=[7, 70], 
                name='CCNN_sghir'):
        super().__init__()    
        self.name = name
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution)) 
        self.fft_end = int(round(frq_band[1] / self.resolution)) + 1
        
        samples = (self.fft_end - self.fft_start) * 2        
        # out_features = samples - kernLength - 1  
        out_features = (samples - (kernLength-1) - 1) + 1 
        filters = 2*Chans

        self.conv1 = L2(nn.Conv2d(1, filters, (Chans, 1), bias=False, padding="valid"), 
                        weight_decay=l2)
        self.bn1   = nn.BatchNorm2d(filters)
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = L2(nn.Conv2d(filters, filters, (1, kernLength), bias=False, padding="valid"),
                        weight_decay=l2)
        self.bn2   = nn.BatchNorm2d(filters)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = L2(nn.Linear(filters * out_features, nb_classes), 
                        weight_decay=l2)

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
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = flatten(x, 1)
        x = self.fc(x)
        return x

