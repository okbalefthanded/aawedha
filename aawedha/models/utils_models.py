from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import load_model as K_load_model
from tensorflow import keras
import tensorflow as tf
import numpy as np


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