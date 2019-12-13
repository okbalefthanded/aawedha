"""
"""
from abc import ABCMeta, abstractmethod
# ?import model
# ?import pipeline

class Process(metaclass=ABCMeta):
    """Process base class

    Attributes
    ----------
    data : Dataset object

    model : model object
        A Keras model to be trained

    pipeline : scikit-learn pipeline object
        the process steps in a pipeline to be run

    params :
        dict of pipeline parameters

    Methods
    -------

    split_set()

    run()
    """
    def __init__(self, data=None, model=None, pipeline=None, params=None):
        self.data = data
        self.model = model
        self.pipeline = pipeline
        self.params = params

    @abstractmethod
    def split_set(self, n_sets=30, split=None):
        """generate n_sets of train/validation/test

        Parameters
        ----------
        n_sets: int
            number of sets to generate

        split: tuple
            size of each set

        Returns
        -------
        """
        pass

    def run(self):
        pass


class SingleSubject(Process):
    """Single subject analysis process
    """


class MultipleSubjects(Process):
    """Multiple subject analysis process
    """
