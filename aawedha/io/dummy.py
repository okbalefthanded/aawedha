'''
    Dummy dataset for experimenting
'''
from aawedha.io.base import DataSet
import numpy as np


class Dummy(DataSet):

    def __init__(self, train_shape=(5, 512, 14, 100),
                 test_shape=(5, 512, 14, 50), nb_classes=5, fs=512):
        """
        """
        super().__init__(title='Dummy', ch_names=[],
                         fs=None, doi='')
        mu, sigma = 0.0, 1.0
        self.epochs = np.random.normal(mu, sigma, train_shape)
        self.y = np.random.randint(low=0, high=nb_classes,
                                   size=(train_shape[0], train_shape[3]))
        self.test_epochs = np.random.normal(mu, sigma, test_shape)
        self.test_y = np.random.randint(low=0, high=nb_classes,
                                        size=(test_shape[0], test_shape[3]))
        self.test_events = []
        #
        self.fs = fs
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_raw(self):
        NotImplementedError

    def generate_set(self):
        """
        """
        val_trials = round(self.epochs.shape[3]*0.8)
        self.x_train = self.epochs[:, :, :, :val_trials]
        self.x_val = self.epochs[:, :, :, val_trials:]
        self.x_test = self.test_epochs
        self.y_train = self.y[:, val_trials]
        self.y_val = self.y[:, val_trials:]
        self.y_test = self.test_y

    def get_path(self):
        NotImplementedError
