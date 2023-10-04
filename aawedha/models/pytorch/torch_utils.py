from thop import profile, clever_format
from torch import nn
import torch

def MaxNorm(tensor, max_value, axis=0):
    eps = 1e-7
    norms = torch.sqrt(torch.sum(torch.square(tensor), axis=axis, keepdims=True))
    desired = torch.clip(norms, 0, max_value)
    return tensor * (desired / (norms + eps))    

# Adapted from braindecode
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, axis=0, **kwargs):
        self.max_norm = max_norm
        self.axis = axis
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = MaxNorm(self.weight.data, self.max_norm, self.axis)
        return super(Conv2dWithConstraint, self).forward(x) 

class LineardWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LineardWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = MaxNorm(self.weight.data, self.max_norm)
        return super(LineardWithConstraint, self).forward(x)    


def count_flops(pth_model):
    """Count PyTorch models FLOPS 

    Parameters
    ----------
    pth_model : TorchModel instance (based on nn.Module)
        a PyTorch model.

    Returns
    -------
    str
        Model's flops

    Raises
    ------
    ValueError
        raises a value error if model's input shape is not predefined.
    """
    if not pth_model.input_shape:
        raise ValueError("Model input shape is not set")
    input_tensor = torch.ones(1, *pth_model.input_shape, device=pth_model.device)
    macs, _ = profile(pth_model.module, inputs=(input_tensor,))
    macs, _ = clever_format([macs, _], "%.3f")
    return macs