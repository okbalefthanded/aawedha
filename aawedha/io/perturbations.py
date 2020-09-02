from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import glob


class Perturbations(DataSet):
    """İşcan Z, Nikulin VV (2018) Steady state visual evoked potential (SSVEP)
    based brain-computer interface (BCI) performance under different
    perturbations. PLoS ONE 13(1): e0191673.
    """

    def __init__(self):
        super().__init__(title='Perturbations',
                         ch_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1',
                                   'C3', 'T7', 'EOGL', 'CP5', 'CP1', 'Pz',
                                   'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                                   'EOGC', 'CP6', 'CP2', 'Cz', 'C4', 'T8',
                                   'EOGR', 'FC6', 'FC2', 'F4', 'F8', 'FP2',
                                   'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7',
                                   'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3',
                                   'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                                   'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8',
                                   'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2',
                                   'AF4', 'AF8', 'BIP1'],
                         fs=1000,
                         doi='https://doi.org/10.1371/journal.pone.0191673')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, mode='', epoch_duration=3,
                 band=[5.0, 45.0], order=6, augment=False):
        ep = epoch_duration
        epoch_duration = np.round(
            np.array(epoch_duration) * self.fs).astype(int)

        X = []
        Y = []
        augmented = 0
        eeg_channels = self._get_eeg_channels()
        onset = 0
        if mode == 'train':
            path = path + '/*mat'
        else:
            path = path + '/*control.mat'
        files_list = glob.glob(path)
        for f in files_list:
            data = loadmat(f)
            x = data['data_matrix'][eeg_channels, :, :].transpose((1, 0, 2))
            if mode == 'train':
                y = data['circle_order'].squeeze()
            else:
                y = data['subject_selection_vector'].squeeze()
            x = bandpass(x, band=band, fs=self.fs, order=order)
            if augment:
                stimulation = 3
                augmented = np.floor(
                    stimulation * self.fs / epoch_duration).astype(int)
                strides = list(np.arange(0, stimulation, ep))
                v = [x[onset + int(s * self.fs):onset + int(s *
                                                            self.fs) + epoch_duration, :, :] for s in strides]
                x = np.concatenate(v, axis=2)
                y = np.tile(y, augmented)
            else:
                x = x[0:epoch_duration, :, :]

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y

    def generate_set(self, load_path=None, epoch=1,
                     band=[5.0, 45.0],
                     order=6, save_folder=None,
                     augment=False):
        '''
        '''
        offline = load_path + '/OFFLINE'
        online = load_path + '/ONLINE'
        self.epochs, self.y = self.load_raw(offline, 'train',
                                            epoch, band,
                                            order, augment
                                            )

        self.test_epochs, self.test_y = self.load_raw(online,
                                                      'test', epoch,
                                                      band, order, augment
                                                      )
        self.subjects = self._get_subjects(n_subjects=24)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events(self.y)
        self.test_events = self._get_events(self.test_y)

        eeg_channels = self._get_eeg_channels()
        self.ch_names = [self.ch_names[i] for i in eeg_channels]
        self.save_set(save_folder)

    def get_path(self):
        NotImplementedError

    def _get_events(self, y):
        '''
        '''
        ev = []
        for i, _ in enumerate(y):
            events = np.empty(y[i].shape, dtype=object)
            for l, _ in enumerate(self.paradigm.frequencies):
                ind = np.where(y[i] == l+1)
                events[ind[0]] = self.paradigm.frequencies[l]
            ev.append(events)
        return ev

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S' + str(s), gender='', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_ON_OFF', control='Sync',
                     stimulation=3000,
                     break_duration=1000, repetition=25,
                     stimuli=4, phrase='',
                     stim_type='ON_OFF',
                     frequencies=['15.', '12.', '8.57', '5.45'],
                     )

    def _get_eeg_channels(self):
        exclude = ['EOGL', 'EOGC', 'EOGR', 'BIP1']
        eeg_channels = [i for i in range(
            64) if self.ch_names[i] not in exclude]
        return eeg_channels