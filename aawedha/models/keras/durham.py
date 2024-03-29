'''
    Durham University models for SSVEP classification


'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten


def D1SCU(Samples=256, nb_classes=5):
    '''
        1DSCU convnet
    '''
    input1 = Input(shape=(1, Samples))
    #
    block1 = Conv1D(16, kernel_size=10, padding='same',
                    kernel_regularizer=l2(0.001), strides=4,
                    input_shape=(1, Samples), use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu',  activity_regularizer=l2(0.001))(block1)
    block1 = MaxPooling1D(pool_size=2, padding='same')(block1)
    block1 = Dropout(0.5)(block1)
    #
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_regularizer=l2(0.001))(flatten)
    softmax = Activation('softmax', activity_regularizer=l2(0.001))(dense)
    #
    return Model(inputs=input1, outputs=softmax, name='1DSCU')


def PodNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5):
    '''
    '''
    input1 = Input(shape=(1, Chans, Samples))
    #
    input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(100, (Chans, 30), padding='valid',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = Dropout(0.5)(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)

    block2 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block1)
    block2 = Dropout(dropoutRate)(block2)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    #
    block3 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block2)
    block3 = Dropout(dropoutRate)(block3)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('relu')(block3)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3)
    #
    block4 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block3)
    block4 = Dropout(dropoutRate)(block4)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('relu')(block4)
    block4 = MaxPooling2D(pool_size=(2, 2), strides=4, padding='same')(block4)
    #
    block5 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block4)
    block5 = BatchNormalization(axis=1)(block5)
    block5 = Activation('relu')(block5)
    block5 = MaxPooling2D(pool_size=(2, 2), strides=4, padding='same')(block5)
    #
    flatten = Flatten()(block5)
    dense = Dense(nb_classes)(flatten)
    #
    softmax = Activation('softmax')(dense)
    return Model(inputs=input1, outputs=softmax, name='PodNet')


def PodNet_small(nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5):
    '''
    '''
    input1 = Input(shape=(1, Chans, Samples))
    #
    input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(100, (Chans, 30), padding='valid',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = Dropout(0.5)(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)

    block2 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block1)
    block2 = Dropout(dropoutRate)(block2)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    #
    block3 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block2)
    block3 = Dropout(dropoutRate)(block3)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('relu')(block3)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block3)
    #
    block4 = Conv2D(100, kernel_size=(1, 30), padding='valid',
                    use_bias=False)(block3)
    block4 = Dropout(dropoutRate)(block4)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('relu')(block4)
    block4 = MaxPooling2D(pool_size=(2, 2), strides=4, padding='same')(block4)
    #
    flatten = Flatten()(block4)
    dense = Dense(nb_classes)(flatten)
    #
    softmax = Activation('softmax')(dense)
    return Model(inputs=input1, outputs=softmax, name='PodNet_Small')
