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

    paradigm :
        dataset experiment info


    Methods
    -------
    load()

    load_raw()

    generate_set()

    get_path()

    """
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        """
         load raw data
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


class Raw:
    """
    Raw EEG data class, used when loading raw data at first

    Attributes
    ----------
    data : numpy array
        raw continuous EEG data as loaded from the file

    events : dict of str
        strings of markers and there respective onset time

    ch_names : list of str
        channel names

    fs : int
        sampling frequency at signal acquisition


    Methods
    -------
    load_raw(path=None, )
        load raw data


    """

    def __init__(self):
        pass

    @abstractmethod
    def load_raw(self, path=None):
        pass






