from torch import nn

def initialize_Glorot_uniform(model):
    """Xavier Glorot uniform weight initialization.        

    Parameters
    ----------
    model: TorchModel instance (sublassing nn.Module)
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def initialize_He_uniform(model):
    """Kaiming He uniform weight initialization.        

    Parameters
    ----------
    model: TorchModel instance (sublassing nn.Module)
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not("BatchNorm" in module.__class__.__name__):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            else: isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)