from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.layers import Dropout, Add, Lambda, DepthwiseConv2D, Input
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def EEGTCNet(nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10, filt=10, 
             dropout=0, activation='relu', F1=4, D=2, kernLength=64, 
             dropout_eeg=0.1):
    
    input1 = Input(shape=(Chans, Samples))
    ##################################################################
    norm = Normalization(axis=(1, 2))(input1)
    reshape = Reshape((1, Chans, Samples))(norm)
    regRate = .25
    numFilters = F1
    F2 = numFilters*D

    EEGNet_sep = EEGNet(input_layer=reshape, F1=F1, kernLength=kernLength, D=D, 
                        Chans=Chans,dropout=dropout_eeg)
    # block2 = Lambda(lambda x: x[:,:,-1,:])(EEGNet_sep)
    sh1, sh2 = EEGNet_sep.shape[1], EEGNet_sep.shape[-1]
    block2 = Reshape((sh1, sh2))(EEGNet_sep)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers,
                     kernel_size=kernel_s, filters=filt, dropout=dropout,
                     activation=activation)
    # out = Lambda(lambda x: x[:,-1,:])(outs)
    out = outs[:,-1,:]
    dense  = Dense(nb_classes, name = 'dense', 
                   kernel_constraint = max_norm(regRate))(out)
    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    softmax      = Activation(activation, name='softmax')(dense)   
    
    return Model(inputs=input1, outputs=softmax, name="EEG-TCNET")

def EEGNet(input_layer=None, F1=4, kernLength=64, D=2, Chans=22, Samples=128, 
            dropout=0.1, fullmodel=False, nb_classes=4):
    F2 = F1*D
    norm_rate =.25
    if fullmodel:
        input1 = Input(shape = (1, Chans, Samples))
        input_layer = input1
    block1 = Conv2D(F1, (1, kernLength), padding = 'same',
                         use_bias = False)(input_layer)
    block1 = BatchNormalization(axis=1)(block1)
    block2 = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (1, 16),
                             use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    block3 = Dropout(dropout)(block3)    

    if fullmodel:
        flatten = Flatten(name = 'flatten')(block3)    
        dense  = Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name = 'softmax')(dense)
        return Model(inputs=input1, outputs=softmax, name="EEGNet")
    else:
        return block3

def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, 
              dropout, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, 
                   activation='linear', padding = 'causal',                   
                   kernel_initializer='he_uniform',
                   data_format='channels_first',)(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1,                    
                   activation='linear', padding = 'causal',
                   kernel_initializer='he_uniform',
                   data_format='channels_first',)(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1, padding='same', 
                      data_format='channels_first',)(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters, kernel_size=kernel_size,                       
                       dilation_rate=2**(i+1),activation='linear',
                       padding = 'causal', kernel_initializer='he_uniform',
                       data_format='channels_first',)(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1),                       
                       activation='linear', padding = 'causal', 
                       kernel_initializer='he_uniform',
                       data_format='channels_first')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out