'''
     Korea University CNN models for  SSVEP classifcation under ambulatory
    environment
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input, Flatten


def KoreaU_CNN_1(nb_classes=5, Samples=120, Chans=8):
    '''
    '''
    input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(8, (8, 1), padding='valid',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = Activation('sigmoid')(block1)
    #
    block2 = Conv2D(8, (1, 11), padding='valid',
                    use_bias=False)(block1)
    block2 = Activation('sigmoid')(block2)
    #
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, activation='sigmoid')(flatten)
    #
    softmax = Activation('sigmoid')(dense)
    return Model(inputs=input1, outputs=softmax)


def KoreaU_CNN_2(nb_classes=5, Samples=120, Chans=8):
    '''
    '''
    input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(8, (8, 1), padding='valid',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = Activation('sigmoid')(block1)
    #
    block2 = Conv2D(8, (1, 11), padding='valid',
                    use_bias=False)(block1)
    block2 = Activation('sigmoid')(block2)
    #
    flatten = Flatten()(block2)
    dense1 = Dense(3, activation='sigmoid')(flatten)
    dense2 = Dense(nb_classes)(dense1)
    #
    softmax = Activation('sigmoid')(dense2)
    return Model(inputs=input1, outputs=softmax)
