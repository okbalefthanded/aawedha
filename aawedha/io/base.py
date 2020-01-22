"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import os
import pickle


class DataSet(metaclass=ABCMeta):
    """DataSet

    Attributes
    ----------
    title : str
        dataset Id

    epochs : ndarray
        epoched EEG data

    y : array
        epochs labels

    events : dict of str
        strings of markers and there respective onset time

    ch_names : list of str
        channel names

    fs : int
        sampling frequency at signal acquisition

    paradigm : object
        dataset experiment info

    subjects : object
        subjects information

    doi : str
        doi of the dataset published paper


    Methods
    -------
    load()

    load_raw()

    generate_set()

    """

    def __init__(self, title='', ch_names=[], fs=None, doi=''):
        self.epochs = []
        self.y = []
        self.events = []
        self.paradigm = None
        self.subjects = []
        self.title = title
        self.ch_names = ch_names
        self.fs = fs
        self.doi = doi

    @abstractmethod
    def load_raw(self):
        pass

    @abstractmethod
    def generate_set(self):
        pass

    @abstractmethod
    def get_path(self):
        pass

    def save_set(self, save_folder=None):
        '''
        '''
        # save dataset
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        if not self.title:
            self.title = 'unnamed_set'

        fname = save_folder + '/' + self.title + '.pkl'
        print(f'Saving dataset {self.title} to destination: {fname}')
        f = open(fname, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        # log if verbose

    def load_set(self, file_name=None, subjects=[], ch=[]):
        '''
        '''
        if os.path.exists(file_name):
            f = open(file_name, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        if subjects:
            data.select_subjects(subjects)
        if ch:
            data.select_channels(ch)
        return data

    def flatten(self):
        '''
        '''
        if type(self.epochs) is list:
            self.epochs = [self._reshape(ep) for ep in self.epochs]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = [self._reshape(ep) for ep in self.test_epochs]
        else:
            self.epochs = self._reshape(self.epochs)
            if hasattr(self, 'test_epochs'):
                self.test_epochs = self._reshape(self.test_epochs)

    def select_subjects(self, subjects=[]):
        '''
        '''
        # TODO : test if equal subjects
        self.epochs = self.epochs[subjects]
        self.y = self.y[subjects]
        if hasattr(self, 'test_epochs'):
            self.test_epochs = self.test_epochs[subjects]
            self.test_y = self.test_y[subjects]

    def select_channels(self, ch=[]):
        '''
        '''
        indexes = [i for i, x in enumerate(self.ch_names) if x in ch]

        self.ch_names = self.ch_names[indexes]

        if type(self.epochs) is list:
            self.epochs = [ep[:, indexes, :] for ep in self.epochs]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = [ep[:, indexes, :] for ep in self.test_epochs]
        else:
            self.epochs = self.epochs[:, :, indexes, :]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = self.test_epochs[:, :, indexes, :]

    def rearrange(self, target_events=[]):
        '''Rearragne dataset by selecting a subset of epochs and their
        labels and events, according to the target events passed.
        Used in CrossSet evaluation

        Parameters
        ----------
        target_events : numpy ndarray of object
            list of events that the selection is based on, if data does not
            contain the same events, the nearest ones in task are selected

        Returns
        -------
        None
        '''
        ind_all = []

        n_subject = len(self.epochs)

        for sbj in range(n_subject):
            ind = np.empty(0)
            for i in range(target_events.size):
                # dataset contain all events in target_events
                ind = np.concatenate((ind, np.where(self.events[sbj] == target_events[i])[0]))
            ind_all.append(ind)
            # tmp.append(np.take(self.epochs[sbj], ind.astype(int), axis=-1))
        self._rearrange(ind_all)

    def recover_dim(self):
        '''
        '''
        channels = len(self.ch_names)
        if type(self.epochs) is list:
            self.epochs = [ep.reshape((ep.shape[0]/ channels, channels, ep.shape[1])) for ep in self.epochs]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = [ep.reshape((ep.shape[0]/ channels, channels, ep.shape[1])) for ep in self.test_epochs]
        else:
            subjects, samples, trials = self.epochs.shape
            self.epochs = self.epochs.reshape((subjects, samples/channels, channels, trials))
            if hasattr(self, 'test_epochs'):
                subjects, samples, trials = self.test_epochs.shape
                self.test_epochs = self.test_epochs.reshape((subjects, samples/channels, channels, trials))

    def get_n_classes(self):
        '''
        '''
        if isinstance(self.y, list):
            return len(np.unique(self.y[0]))
        else:
            return len(np.unique(self.y))

    def _reshape(self, tensor=None):
        '''
        '''
        if tensor.ndim == 4:
            subjects, samples, channels, trials = tensor.shape
            return tensor.reshape((subjects, samples*channels, trials))
        elif tensor.ndim == 3:
            samples, channels, trials = tensor.shape
            return tensor.reshape((samples*channels, trials))

    def _rearrange(self, ind):
        '''
        '''
        # takes only train data for future use in CrossSet
        attrs = ['epochs', 'y', 'events']
        for k in attrs:
            tmp = []
            array = getattr(self, k)
            for sbj in range(len(self.epochs)):
                tmp.append(np.take(array[sbj], ind[sbj].astype(int), axis=-1))
            if isinstance(array, list):
                setattr(self, k, tmp)
            else:
                setattr(self, k, np.array(tmp))
