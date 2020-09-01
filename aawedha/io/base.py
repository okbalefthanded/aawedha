"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod
from aawedha.analysis.utils import isfloat
from scipy.signal import resample_poly
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

    def __str__(self):
        if type(self.epochs) is np.ndarray:
            subjects = len(self.epochs)
            epoch_length = self.epochs.shape[1]
            trials = self.epochs[-1]
        else:
            subjects = 0
            epoch_length = 0.0
            trials = 0

        info = (f'DataSet: {self.title} | <{self.paradigm.title}>',
                f'sampling rate: {self.fs}',
                f'Subjects: {subjects}',
                f'Epoch length: {epoch_length}'
                f'Channels: {self.ch_names}',
                f'Trials:{trials}')
        return '\n'.join(info)

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
                self.test_epochs = [self._reshape(
                    ep) for ep in self.test_epochs]
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
        indexes = self._get_channels(ch)

        self.ch_names = [self.ch_names[ch] for ch in indexes]

        if type(self.epochs) is list:
            self.epochs = [ep[:, indexes, :] for ep in self.epochs]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = [ep[:, indexes, :]
                                    for ep in self.test_epochs]
        else:
            self.epochs = self.epochs[:, :, indexes, :]
            if hasattr(self, 'test_epochs'):
                self.test_epochs = self.test_epochs[:, :, indexes, :]

    def rearrange(self, target_events=[], v=0):
        '''Rearragne dataset by selecting a subset of epochs and their
        labels and events, according to the target events passed.
        Used in CrossSet evaluation

        Parameters
        ----------
        target_events : numpy ndarray of object
            list of events that the selection is based on, if data does not
            contain the same events, the nearest ones in task are selected

        v : float
            interval value for frequencies near target event
        Returns
        -------
        None
        '''
        ind_all = []

        n_subject = len(self.epochs)

        for sbj in range(n_subject):
            ind = np.empty(0)
            tmp = np.empty(0)
            for i in range(target_events.size):
                tmp = np.concatenate(
                    (tmp, np.where(self.events[sbj] == target_events[i])[0]))
                if tmp.size == 0:
                    ind = np.concatenate(
                        (ind, self._get_ind(target_events[i], sbj, v)))
                else:
                    ind = tmp
            ind_all.append(ind)

        self._rearrange(ind_all)

    def update_labels(self, d, v):
        '''
        '''
        k = list(d)
        for sbj in range(len(self.events)):
            r = len(self.events[sbj])

            e = np.array([float(self.events[sbj][i])
                          for i in range(r) if isfloat(self.events[sbj][i])])

            for i in range(len(k)):
                if k[i] == 'idle':
                    idx = np.where(self.events[sbj] == 'idle')[0]
                else:
                    idx = np.logical_and(e > float(k[i])-v, e < float(k[i])+v)
                if isinstance(self.y, list):
                    self.y[sbj][idx] = d[k[i]]
                else:
                    self.y[sbj, idx] = d[k[i]]

    def resample(self, min_rate):
        '''
        '''
        if self.fs == min_rate:
            return
        elif self.fs < min_rate:
            up = self.fs
            down = min_rate
        else:
            up = min_rate
            down = self.fs
        '''
        if isinstance(self.epochs, list):
            self.epochs = [resample_poly(self.epochs[idx], up, down, axis=0)
                           for idx in range(len(self.epochs))]
        else:
            self.epochs = resample_poly(self.epochs, up, down, axis=1)
        '''
        self.epochs = self._resample_array(self.epochs, up, down)
        if hasattr(self, 'test_epochs'):
            self.test_epochs = self._resample_array(self.test_epochs, up, down)

    def recover_dim(self):
        '''
        '''
        channels = len(self.ch_names)
        if type(self.epochs) is list:
            self.epochs = [ep.reshape(
                           (ep.shape[0] / channels, channels, ep.shape[1]))
                           for ep in self.epochs]

            if hasattr(self, 'test_epochs'):
                self.test_epochs = [ep.reshape(
                                    (ep.shape[0] / channels, channels, ep.shape[1]))
                                    for ep in self.test_epochs]
        else:
            subjects, samples, trials = self.epochs.shape
            self.epochs = self.epochs.reshape(
                (subjects, samples/channels, channels, trials))
            if hasattr(self, 'test_epochs'):
                subjects, samples, trials = self.test_epochs.shape
                self.test_epochs = self.test_epochs.reshape(
                    (subjects, samples/channels, channels, trials))

    def get_n_classes(self):
        '''
        '''
        if isinstance(self.y, list):
            return len(np.unique(self.y[0]))
        else:
            return len(np.unique(self.y))

    def _get_channels(self, ch=[]):
        """returns indices of specific channels from channels in dataset

        Parameters
        ----------
        ch : list
            list of channels names

        Returns
        -------
        list
            list of channels indices in ch
        """
        return [i for i, x in enumerate(self.ch_names) if x in ch]

    def _resample_array(self, ndarray, up, down):
        '''
        '''
        if isinstance(ndarray, list):
            ndarray = [resample_poly(ndarray[idx], up, down, axis=0)
                       for idx in range(len(ndarray))]
        else:
            ndarray = resample_poly(ndarray, up, down, axis=1)

        return ndarray

    def _reshape(self, tensor=None):
        '''
        '''
        if tensor.ndim == 4:
            subjects, samples, channels, trials = tensor.shape
            return tensor.reshape((subjects, samples*channels, trials))
        elif tensor.ndim == 3:
            samples, channels, trials = tensor.shape
            return tensor.reshape((samples*channels, trials))

    def _get_ind(self, ev, sbj, v):
        '''
        '''
        rng = np.linspace(-0.25, 1, 25, endpoint=False)
        rng = np.delete(rng, np.where(rng == 0.0)[0])
        tmp = np.empty(0)
        r = len(self.events[sbj])
        e = np.array([float(self.events[sbj][i])
                      for i in range(r) if isfloat(self.events[sbj][i])])

        if isfloat(ev):
            # rg = float(ev) + rng
            # idx = np.logical_or.reduce([e == r for r in rg])
            # tmp = np.concatenate((tmp, np.where(idx == True)[0]))
            idx = np.logical_and(e > float(ev)-v, e < float(ev)+v)
            tmp = np.concatenate((tmp, np.where(idx)[0]))

        return tmp

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
