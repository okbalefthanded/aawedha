from aawedha.loss.focal_loss import FocalLoss
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from holocron.nn import PolyLoss
from ranger21 import Ranger21
from torch import optim
from torch import nn
import torchmetrics

losses = {
    'binary_crossentropy': nn.BCEWithLogitsLoss,
    'sparse_categorical_crossentropy': nn.CrossEntropyLoss,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'focal_loss': FocalLoss,
    'poly_loss': PolyLoss,
    'auc_margin' : AUCMLoss
    }

available_metrics = {
    'accuracy': torchmetrics.Accuracy,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'auc': torchmetrics.AUROC,
    'ece': torchmetrics.CalibrationError,
    'mcc': torchmetrics.MatthewsCorrCoef
    }

custom_opt = {
    'Ranger': Ranger21,
    'PESG' : PESG
}

def get_optimizer(optimizer, opt_params):
    """
    """
    params = {'params': opt_params}
    if isinstance(optimizer, str):
        return _get_optim(optimizer, params)
    elif isinstance(optimizer, list):
        params = {**params, **optimizer[1]}
        return _get_optim(optimizer[0], params)
    else:
        return optimizer   

def _get_optim(opt_id, params):
    available = list(optim.__dict__.keys())
    if opt_id in available:
        return getattr(optim, opt_id)(**params)
    elif opt_id in custom_opt:
        return custom_opt[opt_id](**params)
    else:
        raise ModuleNotFoundError

def get_loss(loss):
    if isinstance(loss, str):
        if loss in list(losses.keys()):
            return losses[loss]()
        else:
            raise ModuleNotFoundError
    elif isinstance(loss, list):
        loss_id = loss[0]
        params = loss[1]
        return losses[loss_id](**params)
    elif isinstance(loss, nn.Module):
        return loss
    else:
        raise NotImplementedError

def get_metrics(metrics):
    selected_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            # quick hack, FIXME
            if metric == 'mcc':
                selected_metrics.append(available_metrics[metric](num_classes=2))    
            else:
                selected_metrics.append(available_metrics[metric]())
        else:
            selected_metrics.append(metric)
    return selected_metrics

def build_scheduler(data_loader, optimizer, scheduler):
    available = list(optim.lr_scheduler.__dict__.keys())
    sched_id = scheduler[0]        
    params = {'optimizer': optimizer, **scheduler[1]}
    if sched_id == 'OneCycleLR':
        params['steps_per_epoch'] = len(data_loader)
    if sched_id in available:
        return getattr(optim.lr_scheduler, sched_id)(**params)
    else:
        ModuleNotFoundError   