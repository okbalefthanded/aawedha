"""Implements:
Pan, Y., Chen, J., Zhang, Y., &#38; Zhang, Y. (2022). An efficient CNN-LSTM 
network with spectral normalization and label smoothing technologies for SSVEP 
frequency recognition. Journal of Neural Engineering 19(5). https://doi.org/10.1088/1741-2552/ac8dc5
"""
from aawedha.trainers.torch_utils import Conv2dWithConstraint
from aawedha.trainers.torchdata import reshape_input
from aawedha.models.utils_models import is_a_loss
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrizations import _SpectralNorm
from torch.nn.utils import parametrize
from torch import flatten
from torch import nn

# conv-bn-prelu-drop
# conv-bn-prelu-drop
# bi lstm
# dense-dense-dense 
class SSVEPNET(nn.Module):
  def __init__(self, nb_classes=4, Samples=256, Chans=8,  dropout_rate=0.5, name='SSVPNET'):
    super().__init__()   
    self.name = name
    out_features = 8 * Chans * ((Samples - 10)//2 + 1)
    
    self.conv1 = spectral_norm(Conv2dWithConstraint(1, 2*Chans, (Chans, 1), bias=False, padding="valid"))
    self.bn1   = nn.BatchNorm2d(2*Chans)
    self.act1  = nn.PReLU()
    self.drop1 = nn.Dropout(dropout_rate)
    self.conv2 = spectral_norm(nn.Conv2d(2*Chans, 4*Chans, (1, 10), stride=2, bias=False, padding="valid"))
    self.bn2   = nn.BatchNorm2d(4*Chans)
    self.act2  = nn.PReLU()
    self.drop2 = nn.Dropout(dropout_rate)
    self.lstm  = nn.LSTM(input_size=4*Chans, hidden_size=4*Chans, num_layers=1, batch_first=True, bidirectional=True)
    self.dense1 = spectral_norm(nn.Linear(out_features, out_features // 10))
    self.act3   = nn.PReLU()
    self.drop3  = nn.Dropout(dropout_rate)
    self.dense2 = spectral_norm(nn.Linear(out_features // 10, out_features // 50))
    self.act4   = nn.PReLU()
    self.drop4  = nn.Dropout(dropout_rate)
    self.dense3 = spectral_norm(nn.Linear(out_features // 50, nb_classes))
    
    self.init_weights()

  def init_weights(self):
    for module in self.modules():
      cls_name = module.__class__.__name__
      if not is_a_loss(module) and not "ModuleDict" in cls_name: # add moduledict test
        if cls_name == "LSTM" or hasattr(module, "weight") or not "PReLU" in cls_name:
          if "Parametrized" in cls_name:
              # re-init for spectral norm parametrized modules, this is a workaround
              # issue #25092: https://github.com/pytorch/pytorch/issues/25092
              parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
              module.reset_parameters()
              parametrize.register_parametrization(module, "weight", _SpectralNorm(module.weight))
          else:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()        
            # else:                       
            #    nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
          if module.bias is not None and isinstance(module.bias, nn.Parameter):
            nn.init.constant_(module.bias, 0)

  def forward(self, x):
    x = reshape_input(x)
    x = self.drop1(self.act1(self.bn1(self.conv1(x))))
    x = self.drop2(self.act2(self.bn2(self.conv2(x))))
    x = x[:,:,-1,:].permute((0,2,1)) # NC1W --> NWC (NLH_in)
    x, _ = self.lstm(x)
    x = flatten(x, 1)
    x = self.drop3(self.act3(self.dense1(x)))
    x = self.drop4(self.act4(self.dense2(x)))
    x = self.dense3(x)
    return x