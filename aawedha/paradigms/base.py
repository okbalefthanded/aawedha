"""
    Base class for paradigms
"""
from abc import ABCMeta, abstractmethod

class Paradigm(metaclass=ABCMeta):
    """
    Paradigms:

    Attributes
    ----------
    title : str
        paradigm title : ERP / Motor Imagery / SSVEP

    control : str
        stimulation mode of paradigm : synchrounous  / asynchrounous

    stimulation : int
        duration of a single stimulation in msec

    break_duration : int
        duration of stimulation pause between two consecutive stimuli in msec

    repetition : int
        number of stimulations per trial (ERP) / session (SSVEP)

    stimuli : int
        number of stimulus presented in the experiment.

    stim_type: str
        stimulus presented to subject (ERP) / type of stimulations used in the experiment (SSVEP) 

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    Methods
    -------
    """

    def __init__(self, title=None, control=None, stimulation=0, 
                    break_duration=0, repetition=0, stimuli=0, stim_type=None, phrase=None):
        self.title = title
        self.control = control
        self.stimulation = stimulation
        self.break_duration = break_duration
        self.repetition = repetition
        self.stimuli = stimuli
        self.stim_type = stim_type
        self.phrase = phrase