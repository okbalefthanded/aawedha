from torchmetrics.classification import MulticlassAccuracy
import torch 

class CategoricalAccuracy(MulticlassAccuracy):
    def __init__(self, task="multiclass", num_classes=None):
        super().__init__(num_classes=num_classes, average="micro")        
        self.task = task
        
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds.argmax(dim=-1), target.argmax(dim=-1))
    
    def compute(self) -> torch.Tensor:
        return super().compute()