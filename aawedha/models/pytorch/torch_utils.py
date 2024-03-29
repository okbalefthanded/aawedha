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