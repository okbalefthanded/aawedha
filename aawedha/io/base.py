"""
    Base class for datasets
"""
from abc import ABCMeta, abstractmethod


class DataSet(metaclass=ABCMeta):
    """DataSet

    parameters
    ----------
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

    def generate_set(self):
        pass

    @abstractmethod
    def get_path(self):
        pass


class Raw:
    """

    """




