"""
    ERP paradigm 

"""
from aawedha.paradigms.base import Paradigm

class ERP(Paradigm):
    """
    Attributes
    ----------
    title : str
        paradigm title : ERP / Motor Imagery / SSVEP

    control : str
        stimulation mode of paradigm : synchrounous  / asynchrounous

    stimulation : int
        duration of a single stimulation in msec

    isi : int
        inter stimulation interval : duration of stimulation pause between two consecutive stimuli in msec

    repetition : int
        number of stimulations per trial

    stimuli : int
        number of stimulus presented in the experiment

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    stim_type : str
        stimulus presented to subject: flash / face / inverted_face ...

    flashing_mode : str
        whether stimuli are presented in a row-column (RC) fashion or single character (SC)

    Methods
    -------
    """
    def __init__(self, title='ERP', control='Sync', stimulation=100, 
                    break_duration=100, repetition=10, stimuli=12, phrase='12345',
                    stim_type='flash', flashing_mode='SC'):
        super(ERP, self).__init__(title, control, stimulation, break_duration, repetition,
                        stimuli, stim_type, phrase)         
        self.flashing_mode = flashing_mode