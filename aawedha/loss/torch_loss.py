from typing import Any, List, Optional, Union
from aawedha.loss import torch_functional as F
from torch import Tensor
from torch import nn
from copy import deepcopy
import torch 


class _Loss(nn.Module):
    def __init__(
        self,
        weight: Optional[Union[float, List[float], Tensor]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # Cast class weights if possible
        self.weight: Optional[Tensor]
        if isinstance(weight, (float, int)):
            self.register_buffer("weight", torch.Tensor([weight, 1 - weight]))
        elif isinstance(weight, list):
            self.register_buffer("weight", torch.Tensor(weight))
        elif isinstance(weight, Tensor):
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ["none", "mean", "sum"]:
            raise NotImplementedError("argument reduction received an incorrect input")
        self.reduction = reduction

    def labels_to_sparse(self, target):
        sparse_targets = deepcopy(target)
        if sparse_targets.ndim > 1:
            if sparse_targets.shape[1] > 1:
                sparse_targets = sparse_targets.argmax(axis=1)
        return sparse_targets

class FocalLoss(_Loss):
    """https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25 , gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        labels = target.float()
        if not (target.size() == x.size()):
            labels = target[...,None].float()
        return F.sigmoid_focal_loss(x, labels, self.alpha, self.gamma, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"


class ComplementCrossEntropy(_Loss):
    """Implements the complement cross entropy loss from
    `"Imbalanced Image Classification with Complement Cross Entropy" <https://arxiv.org/pdf/2009.02189.pdf>`_
    Args:
        gamma (float, optional): smoothing factor
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, gamma: float = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.complement_cross_entropy(x, self.labels_to_sparse(target), 
                                          self.weight, self.ignore_index, 
                                          self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class PolyLoss(_Loss):
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        eps (float, optional): epsilon 1 from the paper
        ignore_index: int = -100,
        reduction: str = 'mean',
    """

    def __init__(
        self,
        *args: Any,
        eps: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.poly_loss(x, self.labels_to_sparse(target), 
                            self.eps, self.weight, self.ignore_index, 
                            self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"