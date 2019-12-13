"""
    SSVEP paradigm

"""
from aawedha.paradigms.base import Paradigm


class SSVEP(Paradigm):
    """
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

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    stim_type : str
        type of stimulations used in the experiment : ON_OFF / Sampled_Sinusoidal /

    frequencies : list
        frequencies presented at each stimuli in Hz

    phase : list
        phase of each frequency in rad

    Methods
    -------
    """

    def __init__(self, title='SSVEP', control='Sync', stimulation=4000,
                 break_duration=4000, repetition=10, stimuli=4, phrase='1234',
                 stim_type='ON_OFF', frequencies=['7.5', '8.57', '10', '12'], phase=None):
        super().__init__(title, control, stimulation, break_duration, repetition,
                         stimuli, stim_type, phrase)
        self.frequencies = frequencies
        self.phase = phase
