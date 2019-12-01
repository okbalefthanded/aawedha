"""
    Hybrid aradigm

"""
# from aawedha.paradigms.base import Paradigm
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.ssvep import SSVEP
import numpy as np


class HybridLARESI(object):

    def __init__(self, title='LARESI_HYBRID', erp=None,
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
                       speller=['1', '2', '3', '4', '5', '6', '7', '8', '9'])

    def _set_SSVEP(self):
        self.ssvep = SSVEP(title='SSVEP_LARESI', control='Async',
                           stimulation=1000,
                           break_duration=2000, repetition=15,
                           stimuli=5, phrase='',
                           stim_type='Sinusoidal', frequencies=['14', '12', '10', '8'],
                           )

    @staticmethod
    def selection_accuracy(events, predictions, y):
        '''
        '''
        if predictions.ndim == 2:
            preds = predictions[:, -1]
        else:
            preds = predictions
        fbk = []
        scr = []
        k = 0
        for i in range(len(events)):
            f, s = HybridLARESI().select_target(preds[i+(k*8):1+i+(k+1)*8], events[i])
            fbk.append(f)
            scr.append(s)
            k += 1
        fbk = np.array(fbk)
        events = np.array(events).flatten()
        idx = y.squeeze() == 1
        acc_selection = np.mean(events[idx] == fbk)

        return acc_selection

    @staticmethod
    def select_target(predictions, events):
        '''
        '''
        commands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        scores = []
        values = set(events)
        for i in range(1, len(values) + 1):
            item_index = np.where(events == i)
            cl_item_output = predictions[item_index]
            score = np.sum(cl_item_output) / len(cl_item_output)
            scores.append(score)
        #
        if scores.count(0) == len(scores):
            feedback_data = '#'
        else:
            feedback_data = int(commands[scores.index(max(scores))])

        return feedback_data, scores.index(max(scores))
