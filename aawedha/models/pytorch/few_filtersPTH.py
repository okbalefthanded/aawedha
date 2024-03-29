from aawedha.models.pytorch.torchmodel import TorchModel
from torch import nn
from torch import flatten
import torch


class SepConv1DPTH(TorchModel):
    """Simple CNN architecture introduced in [1], consisting of
    2 layers, a single input and n_Filters of 1D Separable Convolutions.

    References:
    [1] Alvarado-González, M., Fuentes-Pineda, G., & Cervantes-Ojeda, J. (2021). 
    A few filters are enough: Convolutional neural network for P300 detection. 
    Neurocomputing, 425, 37–52. https://doi.org/10.1016/j.neucom.2020.10.104

    Parameters
    ----------
    Chans : int, optional
        number of EEG channels in data, by default 6
    Samples : int, optional
        EEG trial length in samples, by default 206
    Filters : int, optional
        number of SeparableConv1D filters, by default 32

    Returns
    -------
    PyTorch Module Model instance
    """
    def __init__(self, nb_classes=1, Chans=6, Samples=206, Filters=32, 
                 device="cuda", name="SepCon1DPTH"):
        super().__init__(device=device, name=name)
        # self.pad = nn.ZeroPad2d(padding=(0,0,8,9))
        self.pad = nn.ZeroPad2d(padding=(0,0,5,4))
        self.conv_sep_depth = nn.Conv1d(Chans+9, Filters, 16, bias=True, stride=8, groups=1)
        # self.conv_sep_depth = nn.Conv1d(Filters, Filters, 16, bias=True, stride=8, groups=Filters)
        self.conv_sep_point = nn.Conv1d(Filters, Filters, 1, bias=True)
        self.dense = nn.Linear((Filters * ((Samples // 8)-1)), nb_classes)

        self.initialize_glorot_uniform()
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = torch.tanh(x)
        x = flatten(x,1)     
        x = self.dense(x)      
        return x