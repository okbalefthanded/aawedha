from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
import tensorflow as tf
import tensorflow.keras as keras
from scipy.fft import fft
import numpy as np


def fft_features(epochs=None, fs=512):
    """Calculate FFT features of epochs in the range 3-33 Hz

    Parameters
    ----------
    fs : int
        sampling rate
    epochs : dataset instance
        EEG epoched trials (subjects x samples x channels x trials)
    Returns
    -------
        ndarray
    FFT features in the range of 3-33 Hz 
    (subjects x frequency_points x channels x trials)
    """
    if epochs.ndim == 4:
        subjects, samples, channels, trials = epochs.shape
    elif epochs.ndim == 3:
        subjects = 1
        samples, channels, trials = epochs.shape
    nyquist = fs / 2
    frequencies = np.arange(0, nyquist, fs / samples)
    freq_range = np.logical_and(frequencies >= 3., frequencies < 33.)
    rg = sum(freq_range)
    trials_fft = np.empty((subjects, rg, channels, trials))
    for subj in range(subjects):
        for tr in range(trials):
            bins = fft(epochs[subj, :, :, tr]) / samples
            bins = 2 * np.abs(bins)
            bins = bins[0:len(frequencies)]
            # trials_fft.append(bins[freq_range, :])
            trials_fft[subj, :, :, tr] = bins[freq_range, :]
    return trials_fft


class fft_feat(keras.layers.Layer):
    def __init__(self, samples, idx, **kwargs):
        # self.samples = tf.constant(samples, dtype=tf.float32)
        # self.idx = tf.constant(idx, dtype=tf.bool)
        # self.bins_shape = tf.math.reduce_sum(tf.cast(self.idx, tf.int32), dtype=tf.int32)
        self.samples = samples
        self.idx = idx
        self.bins_shape = np.sum(idx)
        super(fft_feat, self).__init__(**kwargs)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[2]
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for ep in tf.range(batch):
            x = tf.cast(inputs[ep, :, :], tf.complex64)
            x = tf.math.real(tf.signal.fft(x))
            x = tf.math.divide(x, self.samples)
            x = 2 * tf.math.abs(x)
            x = tf.slice(x, [0, 0], [int(self.samples // 2), channels])
            x = tf.boolean_mask(x, self.idx)
            ta = ta.write(ep, x)
        return ta.stack()

    def compute_output_shape(self, input_shape):
        # bins_shape = tf.math.reduce_sum(tf.cast(self.idx, tf.int32))
        return tf.TensorShape([input_shape[0], self.bins_shape, input_shape[2]])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "samples": self.samples, "idx": self.idx,
                "bins": self.bins_shape}


def XU_JIANG(nb_classes=5, Samples=60, Chans=6):
    '''
    '''
    input1 = input1 = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(6, (Chans, 1), padding='valid',
                    input_shape=(1, Chans, Samples))(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=1, padding='valid')(block1)

    block2 = Conv2D(16, (1, Samples // 2), padding='valid')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)

    flatten = Flatten()(block2)
    dense1 = Dense(15, activation='sigmoid')(flatten)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(nb_classes, activation='sigmoid')(dense1)
    softmax = Activation('softmax')(dense2)

    return Model(inputs=input1, outputs=softmax, name='XU_JIANG')


def XU_JIANG_end_to_end(nb_classes=5, Samples=512, Chans=6, fs=512, freq=[3., 33.]):
    """
    """
    ######
    nyquist = fs / 2
    frequencies = np.arange(0, nyquist, fs / Samples)
    idx = np.logical_and(frequencies >= freq[0], frequencies < freq[1])
    bins = np.sum(idx)
    ######
    input1 = Input(shape=(Chans, Samples))
    transpose = tf.keras.layers.Permute((2, 1))(input1)
    fft_ = fft_feat(Samples, idx)(transpose)
    permute = tf.keras.layers.Permute((2, 1))(fft_)
    reshape = tf.keras.layers.Reshape((1, Chans, bins))(permute)
    ####
    block1 = Conv2D(6, (Chans, 1), padding='valid',
                    input_shape=(1, Chans, bins))(reshape)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=1, padding='valid')(block1)

    block2 = Conv2D(16, (1, bins // 2), padding='valid')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('relu')(block2)

    flatten = Flatten()(block2)
    dense1 = Dense(15, activation='sigmoid')(flatten)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(nb_classes, activation='sigmoid')(dense1)
    softmax = Activation('softmax')(dense2)

    return Model(inputs=input1, outputs=softmax, name='XU_JIANG_end_to_end')
