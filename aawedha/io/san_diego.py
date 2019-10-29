from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import glob
import pickle
import os
import gzip


class SanDiego(DataSet):
    '''
        San Diego SSVEP joint frequency and phase modulation dataset [1]
        [1] Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
        A Comparison Study of Canonical Correlation Analysis Based Methods for
        Detecting Steady-State Visual Evoked Potentials,"
        PLoS One, vol.10, no.10, e140703, 2015.
    '''
    def ___init__(self):
        super().__init__(title='San_Diego',
                       ch_names=['PO7','PO3','POz','PO4','PO8','O1','Oz','O2'], 
                       fs=256, 
                       doi = 'http://dx.doi.org/10.1371/journal.pone.0140703'
                       )  

    def load_raw(self, path=None, epoch_duration=1, 
                  band=[5.0, 45.0], order=6):
        list_of_files = glob.glob(path + 's*.mat')
        list_of_files.sort() 

        epoch_duration =  np.round(np.array(epoch_duration) * self.fs).astype(int)
        onset = 39 # onset in samples
        n_subjects = 10               
        X = []
        Y = []

        for subj in range(n_subjects):
            data = loadmat(list_of_files[subj])
            eeg = data['eeg'].transpose((2,1,3,0)) #samples, channels, trials, targets
            eeg = bandpass(eeg, band=band, fs=self.fs, order=order)
            eeg = eeg[onset:onset+epoch_duration, :, :, :]
            samples, channels, blocks, targets = eeg.shape
            y = np.tile(np.arange(1, targets+1), blocks)
            X.append(eeg.reshape((samples, channels, blocks*targets)))
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def generate_set(self, load_path=None, epoch=1, band=[5.0, 45.0], 
                        order=6, save_folder=None):
        self.epochs, self.y = self.load_raw(load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=10)
        self.paradigm = self._get_paradigm()
        
        # save dataset
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        fileName = save_folder + '/san_diego.pkl'
        f = gzip.open(fileName, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()

    def load_set(self, fileName=None):
        """
        """
        if os.path.exists(fileName):
            f = gzip.open(fileName, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S'+str(s),gender='M',age=0, handedness='')
                    for s in range(1, n_subjects+1)]

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_JFPM', stimulation=4000, break_duration=1000, repetition=15,
                    stimilui=12, phrase='',
                    stim_type='ON_OFF', frequencies=[9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 
                    12.25, 14.25, 10.75, 12.75, 14.75], 
                    phase=[0.0, 0.0, 0.0, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 
                           np.pi, np.pi, np.pi, 1.5*np.pi, 1.5*np.pi, 1.5*np.pi] )

    def get_path(self):
        NotImplementedError