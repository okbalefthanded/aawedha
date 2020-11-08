"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod
from aawedha.analysis.utils import isfloat
from aawedha.analysis.preprocess import eeg_epoch
from aawedha.paradigms.subject import Subject
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

    epochs : nd array
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
        self.epochs = None
        self.y = None
        self.events = []
        self.paradigm = None
        self.subjects = []
        self.title = title
        self.ch_names = ch_names
        self.fs = fs
        self.doi = doi
        self.path = '' # keep the path were the final object is saved as pkl

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
                f'Trials: {trials}')
        return '\n'.join(info)

    @abstractmethod
    def load_raw(self):
        """Read and process raw data into structured arrays

        Returns
        -------
        epochs: nd array (subjects x samples x channels x trials)
            epoched EEG data of the whole dataset

        y : nd array (subjects x n_classes)
            datasets labels
        """
        pass

    @abstractmethod
    def generate_set(self):
        """Main method for creating and saving DataSet objects and files:
            - sets train and test (if present) epochs and labels
            - sets dataset information : subjects, paradigm
            - saves DataSet object as a serialized pickle object

        Returns
        -------
        """
        pass

    @abstractmethod
    def get_path(self):
        """Fetch raw dataset files URL

        Returns
        -------

        """
        pass

    @abstractmethod
    def _get_paradigm(self):
        """Get datasets experimental paradigm

        Returns
        -------
        paradigm instance
        """
        pass

    @staticmethod
    def _get_subjects(n_subjects=0):
        """Returns subjects information's as a list of Subject instances

        Parameters
        ----------
        n_subjects : int
            count of subject in a dataset

        Returns
        -------
            list of Subject instances
        """
        return [Subject(id=f'S{s}', gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def save_set(self, save_folder=None, fname=None):
        """Save Dataset instance after creating epochs and attributes in disk
        as a pkl file

        Parameters
        ----------
        save_folder: str
            folder path where to save the DataSet
        fname: str, optional
            saving path for file, specified when different versions of
            DataSet are saved in the same folder
            default: None

        Returns
        -------
        None
        """
        # save dataset
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        if not self.title:
            self.title = 'unnamed_set'
        if fname:
            fname = f'{save_folder}/{fname}.pkl'
        else:
            fname = f'{save_folder}/{self.title}.pkl'

        print(f'Saving dataset {self.title} to destination: {fname}')
        with open(fname, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        # log if verbose

    def load_set(self, file_name=None, subjects=None, ch=None):
        """Load saved DataSet as serialized object
        if subjects are specified, it will return the selected subject(s) data only.
        if ch is specified, a subset of selected channels will be returned

        Parameters
        ----------
        file_name: str
            saved DataSet file path
        subjects: list
            a list of indices, to select subject's Data
        ch : list
            list of channels in str, to select a subset of channels.

        Returns
        -------
            DataSet instance
        """
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
        else:
            raise FileNotFoundError
        if subjects:
            data.select_subjects(subjects)
        if ch:
            _ = data.select_channels(ch)
        return data

    def flatten(self):
        """Transform multichannel EEG into single channel

        Returns
        -------
        None
        """
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

    def select_channels(self, ch=None):
        """Select a subset of channels from specified list of channels

        Parameters
        ----------
        ch: list
            channels names to keep in selection

        Returns
        -------
            list of channels indices in original ch_names
        """
        indexes = self._get_channels(ch)

        self.ch_names = [self.ch_names[ch] for ch in indexes]

        if self.epochs is not None:
            if type(self.epochs) is list:
                self.epochs = [ep[:, indexes, :] for ep in self.epochs]
                if hasattr(self, 'test_epochs'):
                    if self.test_epochs:
                        self.test_epochs = [ep[:, indexes, :] for ep in self.test_epochs]
            else:
                self.epochs = self.epochs[:, :, indexes, :]
                if hasattr(self, 'test_epochs'):
                    if self.test_epochs:
                        self.test_epochs = self.test_epochs[:, :, indexes, :]

        return indexes

    def rearrange(self, target_events=[], v=0):
        '''Rearrange dataset by selecting a subset of epochs and their
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
                tmp = np.concatenate((tmp, np.where(self.events[sbj] == target_events[i])[0]))
                if tmp.size == 0:
                    ind = np.concatenate((ind, self._get_ind(target_events[i], sbj, v)))
                else:
                    ind = tmp
            ind_all.append(ind)

        self._rearrange(ind_all)

    
    def labels_to_dict(self):
        """Attach events to their corresponding labels in a dict
        Parameters
        ----------

        Return
        ------
        dict : keys: events str
               values : labels int
        """
        keys = self.events.flatten().tolist()
        values = self.y.flatten().tolist()
        return dict(zip(keys, values))
        
    
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
                    # idx = np.logical_and(e > float(k[i]) - v, e < float(k[i]) + v)
                    idx = e == float(k[i])
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
                (subjects, samples / channels, channels, trials))
            if hasattr(self, 'test_epochs'):
                subjects, samples, trials = self.test_epochs.shape
                self.test_epochs = self.test_epochs.reshape(
                    (subjects, samples / channels, channels, trials))

    def get_n_classes(self):
        """Get count of classes in DataSet

        Returns
        -------
            int : how many classes the Dataset contains
        """
        if isinstance(self.y, list):
            return len(np.unique(self.y[0]))
        else:
            return len(np.unique(self.y))

    def _get_augmented_cnt(self, raw_signal, epoch, pos, stimulation, slide=0.1, method='divide'):
        """Segment continuous EEG data using an augmentation method

        Parameters
         ----------
        raw_signal: 2d array (samples x channels)
            continuous filtered EEG data

        epoch: array
            epoching window start and finish in samples e.g. [0 , 250]

        pos: array
            stimulation onset in samples

        stimulation: int
            stimulation duration in samples

        slide: float
            sliding window length for 'slide' augmentation method: 0.1 is 100 ms

        method: str
            data augmentation method: 'divide' | 'slide'
            divide: divide the epoch into equal small epochs with no overlap
            slide : divide the epoch into epochs with overlap = epoch-slide

        Returns
        -------
        nd array : samples x channels x (epochs*augmented)
                 augmented = stimulation / epoch if 'divide'
                           = (stimulation - epoch) / slide if 'slide'
                augmented epochs
        """
        v = []
        # stimulation = 5 * self.fs
        if method == 'divide':
            augmented = range(np.floor(stimulation / np.diff(epoch))[0].astype(int))
            v = [eeg_epoch(raw_signal, epoch + np.diff(epoch) * i, pos) for i in augmented]

        elif method == 'slide':
            slide = np.ceil(slide * self.fs).astype(int)
            augmented = range(int((stimulation - np.diff(epoch)) // slide) + 1)
            v = [eeg_epoch(raw_signal, epoch + (slide * i), pos) for i in augmented]

        return v

    def _get_augmented_epoched(self, eeg, epoch, stimulation, onset=0, slide=0.1, method='divide'):
        """Segment epoched EEG data using an augmentation method

        Parameters
        ----------
        eeg: nd array (samples x channels x epochs) | (samples x channels x trials x blocks)
            epoched EEG data

        epoch: int
            epoch duration in seconds e.g. 3 for 3 seconds

        stimulation: int
            stimulation duration in seconds e.g 5 for 5 seconds

        onset: int
            stimulation onset relative to epoch start in seconds e.g. 0.5 is
            500 ms after epoch start

        slide: float
            sliding window length for 'slide' augmentation method: 0.1 is 100 ms

        method: str
            data augmentation method: 'divide' | 'slide'
            divide: divide the epoch into equal small epochs with no overlap
            slide : divide the epoch into epochs with overlap = epoch-slide

        Returns
        -------
            nd array : samples x channels x (epochs*augmented)
                        augmented = stimulation / epoch if 'divide'
                                  = (stimulation - epoch) / slide if 'slide'
            augmented epochs
        """
        # onset = int(0.5 * self.fs)
        epoch_duration = np.round(np.array(epoch) * self.fs).astype(int)
        # stimulation = 5
        v = []

        if method == 'divide':
            strides = range(np.floor(stimulation * self.fs / epoch_duration).astype(int))
            v = [eeg[onset + int(s * self.fs):onset + int(s * self.fs) + epoch_duration] for s in strides]
        elif method == 'slide':
            augmented = range(int((stimulation - epoch) // slide) + 1)
            slide = int(slide * self.fs)
            v = [eeg[onset + slide * s:onset + slide * s + epoch_duration] for s in augmented]

        return v

    def _get_channels(self, ch=None):
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

    def _set_channels(self, channels):
        """Set DataSet channels

        Parameters
        ----------
        channels : list
            new channels

        Returns
        -------

        """
        self.ch_names = channels

    @staticmethod
    def _resample_array(ndarray, up, down):
        '''
        '''
        if isinstance(ndarray, list):
            ndarray = [resample_poly(ndarray[idx], up, down, axis=0)
                       for idx in range(len(ndarray))]
        else:
            ndarray = resample_poly(ndarray, up, down, axis=1)

        return ndarray

    @staticmethod
    def _reshape(tensor=None):
        """Transform multi channels Tensor to a single channel

        Parameters
        ----------
        tensor: nd array (subjects x samples x channels x trials)
                        or (samples x channels x trials)
            multi channels EEG

        Returns
        -------
        nd array : 3D if input is 4D, 2D otherwise
        """

        if tensor.ndim == 4:
            subjects, samples, channels, trials = tensor.shape
            return tensor.reshape((subjects, samples * channels, trials))
        elif tensor.ndim == 3:
            samples, channels, trials = tensor.shape
            return tensor.reshape((samples * channels, trials))

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
            idx = np.logical_and(e > float(ev) - v, e < float(ev) + v)
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
