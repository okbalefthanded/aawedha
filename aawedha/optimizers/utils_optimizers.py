from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.optimizers import get
import tensorflow_addons.optimizers as tfaopt
from madgrad import MadGrad

all_classes_tfa = {'cocob': tfaopt.COCOB,
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
    lib = 'custom'
    try: 
        opt = get(identifier)
        return 'builtin' 
    except ValueError:
        if identifier.lower() in list(tfaopt.__dict__.keys()):
            return 'TFA' # tfa : tensorflow_addons
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

    if opt_lib is 'builtin':
        opt = get(identifier)
        return opt
    elif opt_lib is 'TFA':
        config = {'class_name': str(identifier), 'config': {}}
        opt = deserialize_keras_object(config, module_objects=all_classes_tfa,
                          custom_objects=None, printable_module_name='optimizer')
        return opt
    else:
        config = {'class_name': str(identifier), 'config': {}}
        opt = deserialize_keras_object(config, module_objects=custom_classes,
                          custom_objects=None, printable_module_name='optimizer')
        return opt