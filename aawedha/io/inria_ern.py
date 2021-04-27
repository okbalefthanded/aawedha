from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
import numpy as np
import pandas as pd
import glob


class Inria_ERN(DataSet):
    """
        Feedback Error-Related Negativity dataset [1]

        Reference :
        [1] P. Margaux, M. Emmanuel, D. Sébastien, B. Olivier, and M. Jérémie, Objective and subjective
        evaluation of online error correction during p300-based spelling," Advances in Human-
        Computer Interaction, vol. 2012, p. 4, 2012.
    """

    def __init__(self):
        super().__init__(title='Inria_ERN',
                         ch_names=['Fp1', 'Fp2', 'AF7', 'F3', 'AF4', 'F8', 'F7', 'F5', 'F3', 'F1',
                                   'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                                   'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                                   'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz',
                                   'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
                                   'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2'],
                         fs=200,
                         doi='http://dx.doi.org/10.1155/2012/578295'
                         )
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, epoch_duration=1,
                 band=[1.0, 40.0], order=5):
        """
        """
        list_of_tr_files = sorted(glob.glob(path + '/train/Data_*.csv'))

        list_of_ts_files = glob.glob(path + '/test/Data_*.csv')
        list_of_ts_files.sort()

        n_subjects = 26
        epoch_duration = round(epoch_duration * self.fs)
        channels = len(self.ch_names)
        epochs = 340

        X = self._get_epoched(list_of_tr_files, epoch_duration, band, order)
        X_test = self._get_epoched(
            list_of_ts_files, epoch_duration, band, order)

        train_subjects = 16
        test_subjects = 10

        samples = X[0].shape[0]
        # X = np.array(X).transpose((2,1,0)).reshape((n_subjects, epoch_duration, channels, epochs))
        # X = np.array(X).transpose((2,1,0)).reshape((n_subjects, epoch_duration, channels, epochs), order='F')
        X = np.array(X).reshape((train_subjects, epochs, samples,
                                 channels), order='F').transpose((0, 2, 3, 1))
        X_test = np.array(X_test).reshape(
            (test_subjects, epochs, samples, channels), order='F').transpose((0, 2, 3, 1))

        labels = np.genfromtxt(
            path + '/train//TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]
        Y = labels.reshape((train_subjects, epochs))
        labels_test = np.genfromtxt(
            path + '/test//true_labels.csv', delimiter=',')
        Y_test = labels_test.reshape((test_subjects, epochs))

        # : 4-D array : (subject, samples, channels, epoch)
        return X, Y, X_test, Y_test

    def _get_epoched(self, files_list=None,
                     epoch=1, band=[1., 40.], order=5):
        '''
        '''
        if not files_list:
            return None

        X = []
        for f in files_list:
            sig = np.array(pd.io.parsers.read_csv(f))
            eeg = sig[:, 1:-2]
            trigger = sig[:, -1]
            signal = bandpass(eeg, band, self.fs, order)
            idxFeedback = np.where(trigger == 1)[0]

            for idx in idxFeedback:
                X.append(signal[idx:idx + epoch, :])

            del sig  # saving some RAM

        return X

    def generate_set(self, load_path=None, epoch=1,
                     band=[1.0, 40.0], order=5, 
                     save=True, save_folder=None, fname=None):
        """
        """
        self.epochs, self.y, self.test_epochs, self.test_y = self.load_raw(
            load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=16)
        self.paradigm = self._get_paradigm()
        if save:
            self.save_set(save_folder, fname)

    def get_path(self):
        NotImplementedError

    def _get_events(self):
        NotImplementedError

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S' + str(s), gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def _get_paradigm(self):
        return ERP(title='ERP_ERN', stimulation=60, break_duration=50, repetition=12,
                   phrase='', flashing_mode='RC')
