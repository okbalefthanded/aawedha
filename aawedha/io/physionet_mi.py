from aawedha.io.base import DataSet
from aawedha.paradigms import motor_imagery, subject
from aawedha.analysis.preprocess import bandpass
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import pandas as pd
import glob
import pickle
import os
import gzip


class PhysioNet_MI(DataSet):
    """
        Motor Imagery dataset [1]

        Reference :
        [1]  Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: 
        A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043.
    
        [2]  Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB,
         Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research 
         Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220.

    """
    def __init__(self):
        super().__init__(title='PhysioNet MI', 
                         fs=160,
                         doi='https://doi.org/10.1109/TBME.2004.827072'
                         )
        # adding channels later

    def load_raw(self, path=None, epoch_duration=1, band=[1., 40.], order=6):
        '''
        '''
        subjects = range(1, 110)
        runs = [4, 8 , 12] # imagined hand tasks
        tmin, tmax = -1., 4.
        event_id = dict(left=2, right=3)
        ev_id = dict(T1=2, T2=3)
        pass

    def generate_set(self, load_path=None, epoch=1, band=[1., 40.],
                    order=6, save_folder=None):
        self.epochs, self.y = self.load_raw(load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=109)
        self.paradigm = self._get_paradigm()
        #
        # save dataset
        # save_folder = '/data/physionet'
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        fileName = save_folder + '/physionet_mo.pkl'
        f = gzip.open(fileName, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()

    def load_set(self, fileName=None):
        pass

    def get_path(self):
        NotImplementedError

    def _get_subjects(self, n_subjects=0):
        pass

    def _get_paradigm(self):
        pass

