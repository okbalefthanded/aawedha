#from aawedha.loss.complement_ce import ComplementCrossEntropy
#from aawedha.loss.focal_loss import FocalLoss
#from aawedha.loss.poly_loss import PolyLoss
import aawedha.loss.torch_loss as tl
from aawedha.metrics.torch_metrics import CategoricalAccuracy
from aawedha.models.pytorch.wasamtorch import WASAM
from aawedha.models.pytorch.samtorch import SAM
from aawedha.loss.smooth_loss import SmoothLoss
from aawedha.optimizers.aida import Aida
from aawedha.optimizers.adan import Adan
from aawedha.optimizers.agd import AGD
from lion_pytorch import Lion
from prodigyopt import Prodigy
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from importlib import import_module
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
    'auc_margin' : AUCMLoss,
    "smooth_loss": SmoothLoss,
    # TODO: MSE/MAE
    "mse": nn.MSELoss,
    "mae": nn.L1Loss
    }

available_metrics = {
    'accuracy': torchmetrics.Accuracy,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'auc': torchmetrics.AUROC,
    'ece': torchmetrics.CalibrationError,
    'mcc': torchmetrics.MatthewsCorrCoef,
    'categorical_accuracy': CategoricalAccuracy
    }

custom_opt = {
    'Adan': Adan,
    'Ranger': Ranger21,
    'PESG' : PESG,
    'Lion': Lion,
    'Prodigy': Prodigy,
    'AGD': AGD,
    'Aida': Aida
}

wrapped_opt = {
    'Sam': SAM,
    'Wasam' : WASAM
}

available_callbacks = {
    'none':  NotImplemented,
    }


def get_optimizer(optimizer, opt_params):
    """Create an Optimizer instance from description.

    Parameters
    ----------
    optimizer : str | list
        - optimizer name and parameters
    opt_params : iterable | dict
        Pytorch module (model) parameters to optimize.

    Returns
    -------
    torch.optim instance
        optimizer object
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
    """optimizer creation function

    Parameters
    ----------
    opt_id : str
        optimizer name
    params : dict
        optimizer attributes

    Returns
    -------
    torch.optim instance
        optimizer object

    Raises
    ------
    ModuleNotFoundError
        optimizer name is wrong or it is not implemented.
    """
    available = list(optim.__dict__.keys())    
    if opt_id in available:
        return getattr(optim, opt_id)(**params)
    elif opt_id in custom_opt:
        return custom_opt[opt_id](**params)
    elif opt_id in wrapped_opt:
        return _get_wrapped_optim(opt_id, params)
    else:
        raise ModuleNotFoundError

def _get_wrapped_optim(opt_id, params):
    """Create a wrapped optimizer.
    A wrapped optimizer is a torch.optim object that has a base
    optimizer attribute. eg. SAM. 

    Parameters
    ----------
    opt_id : str
        wrapped optimizer name
    params : dict 
        wrapped optimizer params
    """
    base_opt = params['base_optimizer']
    # params['base_optimizer'] = getattr(opt_module(base_opt), base_opt)
    params['base_optimizer'] = opt_module(base_opt)
    return wrapped_opt[opt_id](**params)

def opt_module(base_opt):
    """Get optimizer module

    Parameters
    ----------
    base_opt : str
        optimizer name

    Returns
    -------
    module
        optimizer class module.
    """
    base_opt_module = None
    if base_opt in list(optim.__dict__.keys()):
        base_opt_module = getattr(optim, base_opt)
    elif base_opt in custom_opt:
        base_opt_module = custom_opt[base_opt]
    return base_opt_module    

def get_loss(loss):
    if isinstance(loss, str):
        if loss in list(losses.keys()):
            return losses[loss]()
        else:
            raise ModuleNotFoundError
    elif isinstance(loss, list):
        loss_id = loss[0]
        params  = loss[1]
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
            selected_metrics.append(available_metrics[metric](task=task, num_classes=classes))
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
        if callback_instance:
            clbks.append(callback_instance)
    return clbks   