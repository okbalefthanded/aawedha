from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from aawedha.models.pytorch.torch_builders import losses
from aawedha.utils.utils import get_device
from timeit import default_timer as timer
from inspect import getfullargspec
from datetime import timedelta
from copy import deepcopy
import tensorflow as tf
import torch


def freeze_model(model, frozen_folder, debug=False):
    """Convert Keras model to a TensorFlow frozen graph
    Parameters
    ----------
    model : Keras model
        Trained model
    frozen_folder : str
        path where to save frozen model
    debug : bool, optional
        if True print name of layers in the model, by default False

    Returns
    -------
    pb_file_name : str
        frozen model file path
    """
    frozen_graph_filename = f"{model.name}_Frozen"
    pb_file_name = f"{frozen_graph_filename}.pb"
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
   
    if debug:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
    
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_folder,
                      name=pb_file_name,
                      as_text=False)

    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_folder,
                      name=f"{frozen_graph_filename}.pbtxt",
                      as_text=True)
    return pb_file_name

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