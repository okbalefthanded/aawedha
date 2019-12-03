'''
    Durham University models for SSVEP classification


'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import backend as K


def D1SCU(Samples=256, nb_classes=5):
    '''
        1DSCU convnet
    '''
    input1 = Input(shape=(1, Samples))
    #
    block1 = Conv1D(16, kernel_size= 10, padding='same', kernel_regularizer=l2(0.001),
                strides=4,input_shape=(1, Samples), use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu',  activity_regularizer=l2(0.001))(block1)
    block1 = MaxPooling1D(pool_size=2, padding='same')(block1)
    block1 = Dropout(0.5)(block1)
    #
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_regularizer=l2(0.001))(flatten)
    softmax = Activation('softmax', activity_regularizer=l2(0.001))(dense)
    #
    return Model(inputs=input1, outputs=softmax)


def PodNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5):
    '''
    In the first Pod, 10×30 convolutional filters are used. 
    . All subsequent convolution filters are
    one-dimensional of size: 1 × 30. 2D Max-pooling operations
    of size: 2 × 2, with stride: 2 and pad: 1 used throughout.
    of drop-out (50%) across convolutions, excluding the final
    convolutional layer in all instances. Each network is optimised
    over 2000 training epochs. Meta-parameters are consistent
    across networks in terms of objective function (stochastic
    gradient descent), learning rate (1e-5), momentum (0.9), and
    batch size (5).
    '''
    input1 = Input(shape=(1, Chans, Samples))
    #
    block1 = Conv2D(Chans, (1, 30), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = Dropout(dropoutRate)(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling1D(pool_size=(2,2), strides=2, padding='same')(block1)
    #
    block2 = Conv1D(1, kernel_size= 30, padding='same',  strides=4, use_bias=False)(block1)
    block2 = Dropout(dropoutRate)(block2)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)
    block2 = MaxPooling1D(pool_size=(2,2), strides=2, padding='same')(block2)
    #
    block3 = Conv1D(1, kernel_size= 30, padding='same',  strides=4, use_bias=False)(block2).
    block3 = Dropout(dropoutRate)(block3)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('relu')(block3)
    block3 = MaxPooling1D(pool_size=(2,2), strides=2, padding='same')(block3)
    #
    block4 = Conv1D(1, kernel_size= 30, padding='same',  strides=4, use_bias=False)(block3)
    block4 = Dropout(dropoutRate)(block4)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('relu')(block4)
    block4 = MaxPooling1D(pool_size=(2,2), strides=2, padding='same')(block4)
    #
    block5 = Conv1D(1, kernel_size= 30, padding='same',  strides=4, use_bias=False)(block4)
    block5 = BatchNormalization(axis=1)(block5)
    block5 = Activation('relu')(block5)
    block5 = MaxPooling1D(pool_size=(2,2), strides=2, padding='same')(block5)
    dense = Dense(nb_classes)(block5)
    #
    softmax = Activation('softmax')(dense)
    #
    return Model(inputs=input1, outputs=softmax)