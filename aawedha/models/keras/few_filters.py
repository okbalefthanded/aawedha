from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import ZeroPadding1D, SeparableConv1D, Input
from tensorflow.keras.layers import Activation, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow import transpose

"""adapted from author's code available at: https://github.com/gibranfp/P300-CNNT
"""

def SepConv1D(nb_classes=1, Chans=6, Samples=206, Filters=32):
    """Simple CNN architecture introduced in [1], consisting of
    2 layers, a single input and n_Filters of 1D Separable Convolutions.

    References:
    [1] Alvarado-González, M., Fuentes-Pineda, G., & Cervantes-Ojeda, J. (2021). 
    A few filters are enough: Convolutional neural network for P300 detection. 
    Neurocomputing, 425, 37–52. https://doi.org/10.1016/j.neucom.2020.10.104

    Parameters
    ----------
    Chans : int, optional
        number of EEG channels in data, by default 6
    Samples : int, optional
        EEG trial length in samples, by default 206
    Filters : int, optional
        number of SeparableConv1D filters, by default 32

    Returns
    -------
    Keras Model instance
    """
    eeg_input    = Input(shape=(Chans, Samples))
    norm         = Normalization(axis=(1, 2))(eeg_input)
    norm         = transpose(norm, perm=[0, 2, 1])
    padded       = ZeroPadding1D(padding=4)(norm)
    padded       = transpose(padded, perm=[0,2,1])
    block1       = SeparableConv1D(Filters, 16, strides=8,
                                 padding='valid',
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 use_bias=True)(padded)
    block1       = Activation('tanh')(block1)
    flatten      = Flatten(name='flatten')(block1)
    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'    
    prediction   = Dense(nb_classes, activation=activation)(flatten)

    return Model(inputs=eeg_input, outputs=prediction, name='SepConv1D')  