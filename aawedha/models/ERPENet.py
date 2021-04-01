"""[summary]
ERPENet[1] model and utils transofmed to suite Aawedha format. Assumes
data.image_format as channels first.

Reference:
[1] A. Ditthapron, N. Banluesombatkul, S. Ketrat, E. Chuangsuwanich, and T. Wilaiprasitporn, 
“Universal Joint Feature Extraction for P300 EEG Classification Using Multi-Task Autoencoder,
” IEEE Access, vol. 7, pp. 68415–68428, 2019.
"""
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.layers import Reshape, UpSampling2D, ZeroPadding2D 
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np



# ERPNet channels
# Cz, C1-6, T7,8, TP7,8, CPz,
# CP1-6, Pz, P1-8, POz, PO3,4,7,8, O1,2 and Oz
# order :
# T7, C5, C3, C1, CZ, C2, C4, C6, T8
# TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8
# P7, P5, P3, P1, PZ, P2, P4, P6, P8
# 0, PO7, 0, PO3, POZ,  PO4, 0, PO8, 0
# 0, 0, 0, O1, OZ, O2, 0, 0, 0  
ch_projection = ['T7', 'C5', 'C3', 'C1','Cz', 'C2', 'C4', 'C6', 'T8',
                 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

projection = [1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1,
              0, 1, 0, 1, 1, 1, 0, 1, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0]



def ERPENet_format(data, channels, projections):
    """Transform epoched EEG into ERPNet projection format
    [subjects x samples x channels x trials] 
            => [subjects x samples x vertical_coordinate x horizontal_coordinate x trials]

    Parameters
    ----------
    data : DataSet instance
        aawedha dataset format

    channels : list of str
        channels names following ERPEnet [1] 35 channels selection

    projections: list of int 0/1
        ERPENet [1] vertical/horizontal coordinate, 1 if channels is present 
        in the grid, 0 otherwise.

    """    
    samples = data.epochs[0].shape[0]
    if isinstance(data.epochs, np.ndarray):
        subjects, trials = data.epochs.shape[0], data.epochs.shape[3]
        tmp_ep = np.zeros((subjects, samples, 5, 9, trials))
    dt_ch = 0
    proj_ch = 0 

    for sbj, sbj_ep in enumerate(data.epochs):    
        tmps = []
        for i in range(sbj_ep.shape[-1]):
            tmp = np.zeros((samples, 45))
            for j, pr in enumerate(projections):
                if pr:
                    if channels[proj_ch] in data.ch_names:
                        tmp[:, j] = sbj_ep[:, dt_ch, i]
                        dt_ch += 1
                    proj_ch += 1  
            dt_ch = 0
            proj_ch = 0                           
            tmp = tmp.reshape((samples, 5, 9))
            tmps.append(tmp)
        if isinstance(data.epochs, list):
            data.epochs[sbj] = np.array(tmps).transpose((1,2,3,0))
        else:
            tmp_ep[sbj] = np.array(tmps).transpose((1,2,3,1))
    
    if isinstance(data.epochs, np.ndarray):
        data.epochs = tmp_ep



def mean_squared_error_ignore_0(y_true, y_pred):
    """ loss function computing MSE of non-blank(!=0) in y_true
    
    Parameters
    ----------
    y_true : TFtensor
        true label
    y_pred : TFtensor
        predicted label
            
    Returns
    -------
    MSE reconstruction error for loss computing
    """
    shape = tf.shape(y_true)
    zeros = tf.zeros(shape)
    loss = K.switch(K.equal(y_true, K.constant(0)), zeros, K.square(y_pred - y_true))
    return K.mean(loss, axis=-1)


def hybrid_LSTM(depth=2, conv_size=16, dense_size=512, input_dim=(100, 5, 9,), dropoutRate=0.2):
    """Autoencoder model builder composes of CNNs and a LSTM

    Parameters
    ----------
    depth : int 
        number of CNN blocks, each has 3 CNN layers with BN and a dropout
    
    conv_size : int
        initial CNN filter size, doubled in each depth level
            
    dense_size :int
        size of latent vector and a number of filters of ConvLSTM2D
            
    input_dim : tuple
        input dimention, should be in (y_spatial,x_spatial,temporal)
            
    dropoutRate : float
        dropout rate used in all nodes

    Returns
    -------
    keras model
    """   


    """Setup"""
    temp_filter = conv_size
    X = Input(shape=input_dim, name='input')    
    model_input = X
    X = Normalization(axis=(2,3))(X)
    X = Reshape((100, 5, 9, 1))(X)

    """Encoder"""
    for i in range(depth):
        for j in range(3):
            # j==0 is first layer(j) of the CNN block(i); apply stride with double filter size
            if j == 0:
                X = TimeDistributed(Conv2D(2*temp_filter, (3, 3), padding='same', strides=(2, 2)), 
                                    name=f"encoder_'{i}{j}_timeConv2D")(X)
            else:
                X = TimeDistributed(Conv2D(temp_filter, (3, 3), padding='same'), 
                                    name=f"encoder_'{i}{j}_timeConv2D")(X)
            X = BatchNormalization(name=f"encoder_'{i}{j}_BN")(X)
            X = LeakyReLU(alpha=0.1, name=f"encoder_'{i}{j}_relu")(X)
            X = Dropout(dropoutRate, name=f"encoder_'{i}{j}_drop")(X)
        temp_filter = int(temp_filter * 2)
    X = TimeDistributed(Flatten())(X)
    X = LSTM(dense_size, recurrent_dropout=dropoutRate,
             return_sequences=False, implementation=2)(X)

    """Latent"""
    latent = X

    """Setup for decoder"""
    X = RepeatVector(100)(X)
    temp_filter = temp_filter//2

    """Decoder"""
    X = LSTM(temp_filter*2*3, recurrent_dropout=dropoutRate,
             return_sequences=True, implementation=2)(X)
    X = Reshape((100, 2, 3, temp_filter))(X)
    for i in range(depth):
        for j in range(3):
            if j == 0:
                X = TimeDistributed(UpSampling2D(
                    (2, 2)), name=f"decoder_{i}{j}_upsampling")(X)
                X = TimeDistributed(ZeroPadding2D(
                    ((1, 0), (1, 0))), name=f"decoder_{i}{j}_padding")(X)
                X = TimeDistributed(Conv2D(temp_filter*2, (3, 3)),
                                    name=f"decoder_{i}{j}_timeConv2D")(X)
            else:
                X = TimeDistributed(Conv2D(temp_filter, (3, 3), padding='same'), 
                                            name=f"decoder_{i}{j}_timeConv2D")(X)
            X = BatchNormalization(name=f"decoder_{i}{j}_BN")(X)
            X = LeakyReLU(alpha=0.1, name=f"decoder_{i}{j}_relu")(X)
            X = Dropout(dropoutRate, name=f"decoder_{i}{j}_drop")(X)
        temp_filter = int(temp_filter / 2)
    X = TimeDistributed(Conv2D(1, (1, 1), padding='same', name='decoder__timeConv2D'))(X)
    X = Reshape((100, 5, 9))(X)
    decoded = X
    X = latent
    X = Dense(1, name='Dense10', activation='sigmoid')(X)
    return Model(inputs=model_input, outputs=[decoded, X])