#from aawedha.loss.complement_ce import ComplementCrossEntropy
#from aawedha.loss.focal_loss import FocalLoss
#from aawedha.loss.poly_loss import PolyLoss
import aawedha.loss.torch_loss as tl
from aawedha.optimizers.adan import Adan
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
#from holocron.nn import PolyLoss
from ranger21 import Ranger21
from torch import optim
from torch import nn
import torchmetrics

losses = {
    'sparse_categorical_crossentropy': nn.CrossEntropyLoss,
    'complement_crossentropy': tl.ComplementCrossEntropy,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'binary_crossentropy': nn.BCEWithLogitsLoss,
    'focal_loss': tl.FocalLoss,
    'poly_loss': tl.PolyLoss,
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
    'Adan': Adan,
    'Ranger': Ranger21,
    'PESG' : PESG
}

available_callbacks = {
    'none':  NotImplemented,
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

def get_metrics(metrics, classes):
    selected_metrics = []
    task = "binary" if classes == 2 else "multiclass"
    
    for metric in metrics:
        if isinstance(metric, str):
            # quick hack, FIXME
            selected_metrics.append(available_metrics[metric](task=task, num_classes=classes))
            '''
            if metric == 'mcc':
                selected_metrics.append(available_metrics[metric](num_classes=2))    
            else:
                selected_metrics.append(available_metrics[metric]())
            '''
        else:
            selected_metrics.append(metric)
    return selected_metrics

def build_scheduler(data_loader, optimizer, scheduler):
    params_args = {'OneCycleLR': 'steps_per_epoch',
                    'CyclicLR': 'step_size_up',
                    'CosineAnnealingWarmRestarts': 'T_0'}
    available = list(optim.lr_scheduler.__dict__.keys())
    sched_id = scheduler[0]
    if sched_id !='cosinewr':   
        sched_id = f"{scheduler[0]}LR"
    else:
        sched_id = "CosineAnnealingWarmRestarts"     
    params = {'optimizer': optimizer, **scheduler[1]}
    if sched_id == 'CyclicLR':
        params[params_args[sched_id]] = len(data_loader) // 2
    else:
        params[params_args[sched_id]] = len(data_loader) 
    '''
    if sched_id == 'OneCycleLR':
        params['steps_per_epoch'] = len(data_loader)
    elif sched_id == '':
        pass
    elif sched_id == 'CosineAnnealingWarmRestarts':
        params['T_0'] = 0
    '''    
    if sched_id in available:
        return getattr(optim.lr_scheduler, sched_id)(**params)
    else:
        ModuleNotFoundError  


def build_callbacks(model, callbacks_list):
    clbks = []
    callback_instance = None
    clbk_id = ""
    for clbk in callbacks_list:
        if isinstance(clbk, list):
            if clbk[0] in available_callbacks:
                clbk_id = clbk[0]
            params = {'model': model, **clbk[1]}
            callback_instance = available_callbacks[clbk_id](**params)
        elif isinstance(clbk, str):
            callback_instance = available_callbacks[clbk](model=model)    
        else:
            callback_instance = clbk
    clbks.append(callback_instance)
    return clbks   