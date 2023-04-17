import torch
import torch.nn as nn
from torch.nn import functional as F

class SmoothLoss(nn.CrossEntropyLoss):
  def __init__(self, alpha=0.6):
    super().__init__()
    assert 0 <= alpha <= 1
    self.alpha = alpha

  def forward(self, input, target):
    sparse_target = target.argmax(dim=-1)
    l_hard = F.cross_entropy(input, sparse_target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
    l_soft = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
    return self.alpha*l_hard + (1-self.alpha)*l_soft

  def __repr__(self):
    return f"{self.__class__.__name__}(alpha={self.alpha}, reduction='{self.reduction}')"