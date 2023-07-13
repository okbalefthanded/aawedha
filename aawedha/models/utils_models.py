from aawedha.models.pytorch.torch_builders import losses
from aawedha.utils.utils import get_device
from timeit import default_timer as timer
from inspect import getfullargspec
from datetime import timedelta
from copy import deepcopy
import tensorflow as tf
import torch


def create_model_from_config(config, optional):
    """Create a Keras model instance from configuration dict.

    Parameters
    ----------
    config : dict
        the model parameters
    optional : dict
        optional parameters: number of channels and length of samples,
        which are speficied after loading data.

    Returns
    -------
    Keras Model instance
    """
    cfg = deepcopy(config)
    mod = __import__(cfg['from'], fromlist=[cfg['name']])
    kwargs = getfullargspec(getattr(mod, cfg['name']).__init__)[0]
    missing_keys = ["nb_classes", "Chans", "Samples", "kernLength"]
    missing_keys.extend(list(optional))
    for key in missing_keys:
        if key not in cfg["parameters"] and key in kwargs:
            cfg["parameters"][key] = optional[key]
    
    # create the instance
    if 'parameters' in cfg.keys():
        params = cfg['parameters']
    else:
        params = {}
    instance = getattr(mod, cfg['name'])(**params)    
    return instance

def model_lib(model_type=None):
    """Retrieve model library from its type

    Parameters
    ----------
    model_type : type
        model instance type, any Keras or Pytorch instance object used
        to create models: Sequential, Functional, Custom.

    Returns
    -------
    str
        library name: Keras or Pytroch
    """
    if "keras" in str(model_type):
        return "keras"
    else:
        return "pytorch"

def load_model(filepath):
    """load classifier saved in filepath, a model can be either a H5 Keras model or
    a Pytorch model.
    
    Parameters
    ----------
    filepath : str
        model's path
    
    Returns
    -------
        - Keras H5 saved model object OR
        - Pytorch model
    """   
            
    if 'h5' in filepath:
        # regular Keras model
        model = tf.keras.models.load_model(filepath)
    elif 'pth' in filepath:
        device = get_device() # available_device()
        if isinstance(device, str):
            device = device.lower()
        # PyTorch Model
        # model = torch.load(filepath) #, map_location=torch.device(device))
        model = torch.load(filepath , map_location=torch.device(device))
        model.set_device(device)
        # model.metrics_to()
    
    return model

def inference_time(model, device, batch=1):
    """Calculate inference time of a given model for a defined tensor.

    Parameters
    ----------
    model : Keras Model | Pytorch Module
        trained model
    device : str
        compute hardware : {CPU | GPU | TPU}
    batch : int, optional
        test data batch size

    Returns
    -------
    str
        inference time in seconds.
    """
    it = None
    model_type = model_lib(type(model)) 
    if model_type == 'keras':
        it =  it_tf(model, batch)
    elif model_type == 'pytorch':
        it = it_pth(model, device, batch)
    return it.total_seconds()

def it_tf(model, batch=1):
    """Calculate inference time of a TensorFlow/Keras Model 

    Parameters
    ----------
    model : TensorFlow/Keras model
        trained model
    batch : int, optional
        test data batch size, by default 1

    Returns
    -------
    str
        inference time in seconds
    """
    tensor = tf.ones((batch, *model.inputs[0].shape[1:]))    
    return elapsed_time(model, tensor)

def it_pth(model, device, batch=1):
    """Calculate inference time of a Pytorch Model 

    Parameters
    ----------
    model : Pytorch model as nn.Module instance.
        trained Pytorch Model
    device : str | torch device
        compute hardware
    batch : int, optional
        test data batch size, by default 1

    Returns
    -------
    str
        inference time in seconds
    """
    model.to(device)
    tensor = torch.ones(batch, *model.input_shape, device=device)    
    return elapsed_time(model, tensor)

def elapsed_time(model, tensor):
    """Estimate elapsed time passed taken by the model to 
    make prediction on tensor.

    Parameters
    ----------
    model : TensorFlow/Keras model | Pytorch Model as nn.Module
        trained model
    tensor : TensorFlow Tensor | Torch Tensor
        a dummy Tensor of ones with shape similar to model input.

    Returns
    -------
    timedelta
        duration of model prediction.
    """
    start = timer()
    pred = model.predict(tensor)
    end = timer()
    return timedelta(seconds=end-start)

def is_a_loss(mod):
    """Test whether a Pytorch nn Module is a loss instance.

    Parameters
    ----------
    mod : nn.Module instance
        a pytorch nn module

    Returns
    -------
    bool
        True if the module is a loss, False otherwise.
    """
    return any([isinstance(mod, loss) for _, loss in losses.items()])