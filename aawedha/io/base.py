"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod
import os
import gzip
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

    def save_set(self, save_folder=None):
        '''
        '''
        # save dataset
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        if not self.title:
            self.title = 'unnamed_set'

        fname = save_folder + '/' + self.title +'.pkl'
        f = gzip.open(fname, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()
        # log if verbose

    def load_set(self, file_name=None):
        '''
        '''
        if os.path.exists(file_name):
            f = gzip.open(file_name, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data  







