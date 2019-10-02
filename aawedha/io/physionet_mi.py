from aawedha.io.base import DataSet
from aawedha.paradigms import motor_imagery, subject
from aawedha.analysis.preprocess import bandpass
from mne import Epochs, pick_types, events_from_annotations
from mne import set_log_level
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

    def load_raw(self, path=None, epoch_duration=1, band=[1., 40.], f_order=6):
        '''
        '''
        set_log_level(verbose=False)
        subjects = range(1, 110)
        runs = [4, 8 , 12] # imagined hand tasks
        #event_id = dict(left=2, right=3)
        ev_id = dict(T1=2, T2=3)     
        subjects = range(1,110)        
        if epoch_duration.__class__ is list:
            # t = [[1., 3.],[1.9, 3.9]]
            t = epoch_duration
        else:
            t = [0, epoch_duration]
        tmin, tmax = -1., 4.
        low_pass, high_pass = band
        epochs = []
        labels = []
        for subj in subjects:         
            raw_names =  ['{f}/S{s:03d}/S{s:03d}R{r:02d}.edf'.format(f=path, s=subj, r=r) for r in runs]
            raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_names])           
            if raw.info['sfreq'] != 160:
                raw.info['sfreq'] = 160  
            # strip channel names of "." characters
            raw.rename_channels(lambda x: x.strip('.'))
            # Apply band-pass filter
            raw.filter(low_pass, high_pass, method='iir', iir_params=dict(order=f_order))
            events, _ = events_from_annotations(raw, event_id=ev_id)
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                exclude='bads')            
            epos = Epochs(raw, events, ev_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)            
            lb = epos.events[:, -1] - 2  
            #
            epo = [epos.copy().crop(tmin=f[0], tmax=f[1]) for f in t]
            epo = np.vstack(epo)
            epo = np.transpose(epo, (2, 1, 0))
            epochs.append(epo)
            labels.append(np.repeat(lb, 2))
        
        self.ch_names = raw.info['ch_names']
        return epochs, labels # list of epochs: len(epochs) = subjects, epochs: samples, channels trials

    def generate_set(self, load_path=None, epoch=1, band=[1., 40.],
                    order=6, save_folder=None):
        self.epochs, self.y = self.load_raw(load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=109)
        self.paradigm = self._get_paradigm()
        
        # save dataset
        # save_folder = '/data/physionet'
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        fileName = save_folder + '/physionet_mi.pkl'
        f = gzip.open(fileName, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()

    def load_set(self, fileName=None):
        if os.path.exists(fileName):
            f = gzip.open(fileName, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data  

    def get_path(self):
        NotImplementedError

    def _get_subjects(self, n_subjects=0):
        return [subject.Subject(id='S'+str(s),gender=None, age=0, handedness='')
                    for s in range(1, n_subjects+1)]

    def _get_paradigm(self):
        return None

