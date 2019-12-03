'''
    Durham University models for SSVEP classification


'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling2D
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