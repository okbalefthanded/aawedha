from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten


def XU_JIANG(nb_classes=5, Samples=60, Chans=6):
    '''
    '''
    input1 = input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(6, (Chans, 1), padding='valid',
                    input_shape=(1, Chans, Samples))(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=1, padding='valid')(block1)

    block2 = Conv2D(16, (1, 30), padding='valid')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)

    flatten = Flatten()(block2)
    dense1 = Dense(15, activation='sigmoid')(flatten)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(nb_classes, activation='sigmoid')(dense1)
    softmax = Activation('softmax')(dense2)

    return Model(inputs=input1, outputs=softmax, name='XU_JIANG')