"""
    Motor Imagery (SMR) paradigm
"""

from aawedha.paradigms.base import Paradigm


class MotorImagery(Paradigm):
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
        number of stimulations per trial

    stimuli : int
        number of stimulus presented in the experiment

    phrase : str
        sequence of characters to be spelled by the subject during the experiments

    stim_type : str
        stimulus presented to subject: flash / face / inverted_face ...


    Methods
    -------
    """

    def __init__(self, title='MotorImagery',
                 control='Sync',
                 stimulation=3000,
                 break_duration=2000,
                 repetition=72,
                 stimuli=4,
                 phrase=None,
                 stim_type='Arrow Cue/Beep'
                 ):
        super().__init__(title, control, stimulation,
                         break_duration, repetition,
                         stimuli, stim_type, phrase)
