import torch.optim as optim
import torch.nn as nn
import torchmetrics
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from ranger21 import Ranger21


losses = {
    'binary_crossentropy': nn.BCEWithLogitsLoss,
    'sparse_categorical_crossentropy': nn.CrossEntropyLoss,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'auc_margin' : AUCMLoss
            }

available_metrics = {
    'accuracy': torchmetrics.Accuracy,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'auc': torchmetrics.AUROC
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
    if loss in list(losses.keys()):
        return losses[loss]()
    else:
        raise ModuleNotFoundError

def get_metrics(metrics):
    selected_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
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