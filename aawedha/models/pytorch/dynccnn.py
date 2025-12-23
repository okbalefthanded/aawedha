from aawedha.models.pytorch.torchdata import reshape_input
from aawedha.models.utils_models import is_a_loss
from aawedha.layers.dynamicconv import DynamicConv
from aawedha.models.pytorch.eegtcnetpth import TemporalConvNet
from scipy.signal import butter, filtfilt
from torchlayers.regularization import L2
from torch.nn.utils.parametrizations import spectral_norm
from torch import flatten
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class DynCCNN(nn.Module):

    def __init__(self, nb_classes=4, Chans=8,  dropout_rate=0.25, kernLength=10,
                 fs=512, resolution=0.293, l2=0.0001, frq_band=[7, 70], dilate=3,
                 K=4, ratio=4, temperature=30, name='DynCCNN'):
        super().__init__()    
        self.name = name
        self.fs = fs
        self.resolution = resolution
        self.nfft  = round(fs / resolution)
        self.fft_start = int(round(frq_band[0] / self.resolution)) 
        self.fft_end   = int(round(frq_band[1] / self.resolution)) + 1
        dilation = (1, dilate) 
        samples = (self.fft_end - self.fft_start) * 2        
        out_features = samples - dilation[1]*(kernLength-1) 
        filters = 2*Chans

        self.conv1 = DynamicConv(1, filters, (Chans, 1), stride=1, padding="valid",
                                    grounps=1, bias=False ,K=K,
                                    temprature=temperature, ratio=1, init_weight=True)
                                   
        self.ln1   = nn.BatchNorm2d(filters) # nn.LayerNorm([filters, 1, samples])
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = DynamicConv(filters, filters, (1, 10), stride=1, padding="valid",
                                    dilation=dilation, grounps=1, bias=False ,K=K,
                                    temprature=temperature, ratio=ratio, init_weight=True)
                                   
        self.ln2   = nn.BatchNorm2d(filters) # nn.LayerNorm([filters, 1, out_features])
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc    = L2(nn.Linear(filters*out_features, nb_classes), weight_decay=l2)

        self.init_weights()
            
    def init_weights(self):
        for module in self.modules():
            cls_name = module.__class__.__name__
            if not is_a_loss(module) and not "ModuleDict" in cls_name:  
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
    

class FBDynCCNN(nn.Module):
  def __init__(self, fs=256, nb_classes=12, n_subbands=3, 
               band=[8, 80], models=None, name="FB-DynCCNN"):
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