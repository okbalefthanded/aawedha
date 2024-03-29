from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
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
    missing_keys = ["Chans", "Samples"]
    for key in missing_keys:
        if key not in cfg["parameters"]:
            cfg["parameters"][key] = optional[key]

    mod = __import__(cfg['from'], fromlist=[cfg['name']])
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
        # device = available_device()
        # PyTorch Model
        model = torch.load(filepath) #, map_location=torch.device(device))
        # model.set_device(device)
    return model

