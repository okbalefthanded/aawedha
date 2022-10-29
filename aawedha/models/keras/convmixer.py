from tensorflow.keras.layers.preprocessing import Normalization
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow import keras

"""Based on Keras examples documentation:
    https://keras.io/examples/vision/convmixer/
"""

def activation_block(x, activation="gelu"):
    x = layers.Activation(activation)(x)
    return layers.BatchNormalization(axis=1)(x) # for channel first

def conv_stem(x, filters: int, patch_size: int, activation: str):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x, activation)

def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", 
                                depthwise_constraint=max_norm(1.))(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x

def ConvMixer(Chans=15, Samples=205, filters=256, depth=8, kernel_size=5, patch_size=2,
              nb_classes=2, activation="gelu", norm_rate=0.25):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((Chans, Samples))
    norm = Normalization(axis=(1, 2))(inputs)
    reshape = Reshape((1, Chans, Samples))(norm)

    # Extract patch embeddings.
    x = conv_stem(reshape, filters, patch_size, activation)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    dense = layers.Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    outputs = Activation(activation, name='softmax')(dense)

    return keras.Model(inputs, outputs)
