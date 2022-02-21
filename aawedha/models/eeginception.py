from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import Dense, Activation, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Concatenate
from aawedha.layers.softpool import SoftPooling2D

def conv_block(tensor, conv_type="Conv2D", filters=8, kernel=(1, 64), pad="same",
               activation='elu', dropout_rate=0.2):
    if conv_type == "Conv2D":
        tensor = Conv2D(filters, kernel, padding=pad)(tensor)
    else:
        tensor = DepthwiseConv2D(kernel, padding=pad, depth_multiplier=filters)(tensor)
    tensor = BatchNormalization(axis=1)(tensor)
    tensor = Activation(activation)(tensor)
    tensor = Dropout(dropout_rate)(tensor)
    return tensor


def EEGInception(nb_classes=1, Chans=15, Samples=205, activation='elu', pooling='avg'):
    if pooling == 'avg':
        pool = AveragePooling2D
    elif pooling== 'soft':
        pool = SoftPooling2D
    
    input1 = Input(shape=(Chans, Samples))
    ##################################################################
    norm = Normalization(axis=(1, 2))(input1)
    reshape = Reshape((1, Chans, Samples))(norm)
    ##################################################################
    # Inception Module 1
    c1 = conv_block(reshape, "Conv2D", 8, (1, 64), "same")
    d1 = conv_block(c1, "Depth2D", 2, (8, 1), "valid")
    c2 = conv_block(reshape, "Conv2D", 8, (1, 32), "same")
    d2 = conv_block(c2, "Depth2D", 2, (8, 1), "valid")
    c3 = conv_block(reshape, "Conv2D", 8, (1, 16), "same")
    d3 = conv_block(c3, "Depth2D", 2, (8, 1), "valid")
    n1 = Concatenate(axis=1)([d1, d2, d3])
    a1 = pool((1, 4))(n1)
    # Inception Module 2
    c4 = conv_block(a1, "Conv2D", 8, (1, 16), "same")
    c5 = conv_block(a1, "Conv2D", 8, (1, 8), "same")
    c6 = conv_block(a1, "Conv2D", 8, (1, 4), "same")
    n2 = Concatenate(axis=1)([c4, c5, c6])
    a2 = pool((1, 2))(n2)
    # Output Module 3
    c7 = conv_block(a2, "Conv2D", 12, (1, 8), "same")
    a3 = pool((1, 2))(c7)
    c8 = conv_block(a3, "Conv2D", 6, (1, 4), "same")
    a4 = pool((1, 2))(c8)
    flatten = Flatten()(a4)

    dense = Dense(nb_classes)(flatten)
    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    softmax = Activation(activation)(dense)
    return Model(inputs=input1, outputs=softmax, name='EEGInception')