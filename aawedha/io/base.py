"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod


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
    def load_set(self):
        """
         load saved dataset
        :parameter:

        :return:
        Continuous EEG data object
        """
        pass

    @abstractmethod
    def load_raw(self):
        pass

    def generate_set(self):
        pass

    @abstractmethod
    def get_path(self):
        pass







