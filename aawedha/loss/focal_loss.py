from torchvision.ops import sigmoid_focal_loss
from typing import Any, List, Optional, Union
from torch import Tensor
from torch import nn
import torch 

# base on:
# https://github.com/frgfm/Holocron/blob/main/holocron/nn/modules/loss.py
# https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
class _Loss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ["none", "mean", "sum"]:
            raise NotImplementedError("argument reduction received an incorrect input")
        self.reduction = reduction

class FocalLoss(_Loss):
    """
    """
    def __init__(self, alpha: float = 0.25 , gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        labels = target.float()
        if not (target.size() == x.size()):
            labels = torch.nn.functional.one_hot(target).float()        
        return sigmoid_focal_loss(x, labels, self.alpha, self.gamma, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"