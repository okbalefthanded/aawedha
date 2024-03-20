from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.optimizers import get
from aawedha.io.io_utils import load_yaml
from madgrad import MadGrad

import tensorflow_addons.optimizers as tfaopt

all_classes_tfa = {'adabelief': tfaopt.AdaBelief,
                    'adamw': tfaopt.AdamW,
                  'cocob': tfaopt.COCOB,
                  'conditional_gradient': tfaopt.ConditionalGradient,
                  'lamb' : tfaopt.LAMB,
                  'novograd': tfaopt.NovoGrad,
                  'proximal_adagrad': tfaopt.ProximalAdagrad,
                  'radam': tfaopt.RectifiedAdam,
                  'yogi': tfaopt.Yogi}

custom_classes = {'madgrad': MadGrad}

def optimizer_lib(identifier):
    """Returns optimizer's library: builtin-Kears, Tensorflow Addons, Custom.
    
    Parameter
    --------
        identifier : str
            optimizer name
    Returns
    -------
        str : optimizer library
    """
    opt_id = identifier
    if isinstance(identifier, list):
        opt_id = identifier[0]
    
    lib = 'custom'
    try:
        _ = get(opt_id)
        return 'builtin'
    except ValueError:
        if opt_id.lower() in list(all_classes_tfa.keys()):
            return 'TFA' # tfa : tensorflow_addons
        else:
            NotImplementedError
    return lib


def get_optimizer(identifier, opt_lib=None):
    """Returns optimizer instance from identifier

    Parameter
    --------
        identifier : str
            optimizer name
    Returns
    -------
        Keras optimizer instace with default values.
    """
    if not opt_lib:
        opt_lib = optimizer_lib(identifier)

    if isinstance(identifier, list):
        opt_id = identifier[0]
        conf = identifier[1]
    else:
        opt_id = identifier
        conf = {}
    
    config = {'class_name': str(opt_id).lower(), 'config': conf}

    if opt_lib == 'builtin':
        opt = get(identifier)
        return opt
    elif opt_lib == 'TFA':
        opt = deserialize_keras_object(config, module_objects=all_classes_tfa,
                          custom_objects=None, printable_module_name='optimizer')
        return opt
    else:
        opt = deserialize_keras_object(config, module_objects=custom_classes,
                          custom_objects=None, printable_module_name='optimizer')
        return opt
    

def config_to_description(cfg_path):
    """convert yaml configuration entries to a list of optimizer
    description for evaluation.

    Parameters
    ----------
    cfg_path : str
        yaml configuration file path.

    Returns
    -------
    list
        list of description with list[0] is the optimizer name
        and list[1] is a dict of optimizer params.
    """
    cfg = load_yaml(cfg_path)
    # return [cfg["name"], cfg["params"]]
    return {cfg["name"]: cfg["params"]}