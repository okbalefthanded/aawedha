'''
    Dummy dataset for experimenting
'''
from aawedha.io.base import DataSet
from tensorflow.keras.utils import to_categorical
import numpy as np


class Dummy(DataSet):

    def __init__(self, train_shape=(5, 512, 14, 100),
                 test_shape=(5, 512, 14, 50), nb_classes=5):
        '''
        '''
        super().__init__(title='Dummy', ch_names=[],
                         fs=None, doi='')
        mu, sigma = 0.0, 1.0
        self.epochs = np.random.normal(mu, sigma, train_shape)
        self.y = np.random.randint(low=0, high=nb_classes, size=train_shape[3])
        self.test_epochs = np.random.normal(mu, sigma, test_shape)
        self.test_y = np.random.randint(low=0, high=nb_classes, size=test_shape[3])
        self.test_events = []
        #
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_raw(self):
        NotImplementedError

    def generate_set(self):
        '''
        '''
        kernels = 1
        val_trials = round(self.epochs.shape[3]*0.8)
        sbj, samples, channels, trials = self.epochs.shape
        self.x_train = self.epochs[:, :, :, :val_trials].transpose(
            (0, 3, 2, 1)).reshape(
                (sbj, val_trials, kernels, channels, samples))
        self.x_val = self.epochs[:, :, :, val_trials:].transpose(
            (0, 3, 2, 1)).reshape(
                (sbj, trials-val_trials, kernels, channels, samples))
        trials = self.test_epochs.shape[3]
        self.x_test = self.test_epochs.transpose((0, 3, 2, 1)).reshape(
            (sbj, trials, kernels, channels, samples))
        self.y_train = to_categorical(self.y[:val_trials])
        self.y_val = to_categorical(self.y[val_trials:])
        self.y_test = to_categorical(self.test_y)

    def get_path(self):
        NotImplementedError
