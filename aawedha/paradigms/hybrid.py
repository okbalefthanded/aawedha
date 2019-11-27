"""
    Hybrid aradigm 

"""
# from aawedha.paradigms.base import Paradigm
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.ssvep import SSVEP

class HybridLARESI(object):

    def __init__(self, title='LARESI_HYBRID', erp = None,
                 ssvep=None, mode='calib' 
                 ):
        
        self.title = title
        self.experiment_mode = mode
        if erp:
            self.ERP = erp
        else:
            self._set_ERP()
        if ssvep:
            self.SSVEP = ssvep
        else:
            self._set_SSVEP()

    def _set_ERP(self):
        self.erp = ERP(title='ERP_LARESI', stimulation=100, 
                   break_duration=50, repetition=5,
                   phrase='123456789', 
                   flashing_mode='SC',
                   speller=['1','2','3','4','5','6','7','8','9'])

    def _set_SSVEP(self):
        self.ssvep = SSVEP(title='SSVEP_LARESI', control='Async',
                    stimulation=1000, 
                    break_duration=2000, repetition=15,
                    stimuli=5, phrase='',
                    stim_type='Sinusoidal', frequencies=['14','12','10','8'], 
                    )