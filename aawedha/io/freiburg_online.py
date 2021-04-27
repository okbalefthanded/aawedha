from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from scipy.io import loadmat
import numpy as np
import glob


class FreiburgOnline(DataSet):
    '''
        Freiburg ERP online speller dataset [1]

        Reference:
        [1] Hübner D, Verhoeven T, Schmid K, Müller K-R, Tangermann M and Kindermans P-J.
        Learning from Label Proportions in Brain-Computer Interfaces: Online Unsupervised
        Learning with Guarantees. PLOS ONE. 2017
    '''

    def __init__(self):
        super().__init__(title='Freiburg_ERP_online',
                         ch_names=['C3', 'C4', 'CP1', 'CP2',
                                   'CP5', 'CP6', 'Cz', 'F10',
                                   'F3', 'F4', 'F7', 'F8',
                                   'F9', 'FC1', 'FC2', 'FC5',
                                   'FC6', 'Fp1', 'Fp2', 'Fz',
                                   'O1', 'O2', 'P10', 'P3',
                                   'P4', 'P7', 'P8', 'P9',
                                   'Pz', 'T7', 'T8'],
                         fs=100,
                         doi='https://doi.org/10.1371/journal.pone.0175856'
                         )

    def load_raw(self, path=None):
        '''
        '''
        files_list = sorted(glob.glob(path + '/S*.mat'))
        n_subjects = 13
        X = []
        Y = []
        for subj in range(n_subjects):
            data = loadmat(files_list[subj])
            X.append(data['epo']['x'][0][0][20:, :, :])
            Y.append(data['epo']['y'][0][0].argmin(axis=0))
            del data

        samples, channels, trials = X[0].shape
        X = np.array(X).reshape(
            (n_subjects, samples, channels, trials), order='F')
        Y = np.array(Y).reshape((n_subjects, trials), order='F')

        return X, Y

    def generate_set(self, load_path=None,
                     save=True, save_folder=None,
                     fname=None):
        '''
        '''
        self.epochs, self.y = self.load_raw(load_path)
        self.subjects = self._get_subjects(n_subjects=13)
        self.paradigm = self._get_paradigm()
        if save:
            self.save_set(save_folder)

    def _get_subjects(self, n_subjects=0):
        '''
        '''
        return [Subject(id='S' + str(s), gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def _get_paradigm(self):
        '''
        '''
        return ERP(title='ERP_FRIEBURG', stimulation=100,
                   break_duration=150, repetition=68,
                   phrase='Franzy jagt im komplett verwahrlosten Taxi quer durch Freiburg',
                   flashing_mode='SC',
                   speller=[])

    def get_path(self):
        NotImplementedError
