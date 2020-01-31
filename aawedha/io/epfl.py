from aawedha.analysis.preprocess import bandpass, eeg_epoch
from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from scipy.io import loadmat
from datetime import datetime
import numpy as np
import glob


class EPFL(DataSet):
    '''
        EPFL Image speller Dataset [1]

        Reference:
        Ulrich Hoffmann, Jean-Marc Vesin, Karin Diserens, and Touradj Ebrahimi.
        An efficient P300-based brain-compuer interface for disabled subjects.
        Journal of Neuroscience Methods, 2007

    '''

    def __init__(self):
        super().__init__(title='EPFL_Image_Speller',
                         ch_names=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7',
                                   'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3',
                                   'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6',
                                   'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8',
                                   'AF4', 'Fp2', 'Fz', 'Cz', 'MA1', 'MA2'],
                         fs=512,
                         doi='https://doi.org/10.1016/j.jneumeth.2007.03.005')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, epoch=[0., .7],
                 band=[1, 10], order=2):
        '''
        '''
        subjects = 9
        sessions = range(1, 5)
        # epochs = []
        # y = []
        events = []
        events_test = []
        target = []
        X = []
        Y = []
        X_test = []
        Y_test = []

        for sbj in range(1, subjects + 1):
            if sbj == 5:
                continue
            raw_names = ['{f}/subject{s}/session{r}/'.format(f=path, s=sbj, r=r) for r in sessions]
            epochs = []
            y = []
            stims = []
            # train sessions
            for session in range(len(raw_names)-1):
                files = glob.glob(raw_names[session] + '*.mat')
                ep, yy, trg, stm = self._load_session(files, epoch, band, order)  # subject, session, runs
                epochs.append(ep)
                y.append(yy)
                target.append(trg)
                stims.append(stm)
            # test session
            test_files = glob.glob(raw_names[-1] + '*.mat')
            test_epochs, test_y, test_target, stm = self._load_session(test_files, epoch, band, order)

            epochs = np.concatenate(epochs, axis=-1)  # subject, sessions, runs
            y = np.concatenate(y, axis=-1)

            X.append(epochs)
            Y.append(y)
            X_test.append(test_epochs)
            Y_test.append(test_y)
            events.append(np.concatenate(stims, axis=-1))
            events_test.append(stm)

        #
        self.events = events
        self.test_events = events_test
        return X, Y, X_test, Y_test

    def generate_set(self, load_path=None,
                     save_folder=None,
                     epoch=[0., 0.7],
                     band=[1, 10],
                     order=2):
        '''
        '''
        self.epochs, self.y, self.test_epochs, self.test_y = self.load_raw(
            load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=9)
        self.paradigm = self._get_paradigm()
        self.save_set(save_folder)

    def _load_session(self, files, epoch, band, order):
        '''
        '''
        epochs = []
        y = []
        target = []
        stims = []
        for run in range(len(files)):
            data = loadmat(files[run])
            ep, y_tmp, trg, stim = self._get_epochs(data, epoch, band, order)
            epochs.append(ep)
            y.append(y_tmp)
            target.append(trg)
            stims.append(stim)

        # epochs = np.array(epochs)
        epochs = np.concatenate(epochs, axis=-1)
        # y = np.array(y)
        y = np.concatenate(y, axis=-1)
        target = np.array(target)
        stims = np.concatenate(stims, axis=-1)
        return epochs, y, target, stims

    def _get_epochs(self, data, epoch, band, order):
        '''
        '''
        original_fs = 2048
        decimation = int(original_fs / self.fs)
        epoch_length = np.round(np.array(epoch) * self.fs).astype(int)
        # following MATLAB code
        signal = data['data']
        events = data['events']
        stimuli = data['stimuli'].squeeze()
        target = data['target'].item()
        #
        ev = []
        for eventi in events:
            ev.append(datetime(*eventi.astype(int), int(eventi[-1] * 1e3) % 1000 * 1000))

        pos = []
        n_trials = len(stimuli)
        for j in range(n_trials):
            delta_seconds = (ev[j] - ev[0]).total_seconds()
            delta_indices = int(delta_seconds * self.fs)
            # has to add an offset
            pos.append(delta_indices + int(0.4 * self.fs))

        eeg_channels = range(32)
        ref_ch = [7, 24]
        ref = np.mean(signal[ref_ch, :], axis=0)
        signal -= ref
        signal = signal[eeg_channels, :]
        signal = bandpass(signal.T, band, original_fs, order)
        signal = signal[::decimation, :]
        epochs = eeg_epoch(signal, epoch_length, pos)
        y = np.zeros(n_trials)
        y[stimuli == target] = 1

        return epochs, y, target, stimuli

    def _get_subjects(self, n_subjects=0):
        '''
        '''
        s = []
        disabled_subjects = 4
        disabled_gender = ['M', 'M', 'M', 'F']
        disabled_age = [56, 51, 47, 33]
        disabled_condition = ['Cerebral palsy',
                              'Multiple sclerosis',
                              'Late-stage amyotrophic lateral sclerosis',
                              'Traumatic brain and spinal-cord injury, C4 level']

        for sbj in range(n_subjects):
            if sbj < disabled_subjects:
                s.append(Subject(id='S' + str(sbj),
                                 gender=disabled_gender[sbj],
                                 age=disabled_age[sbj],
                                 handedness='',
                                 condition=disabled_condition[sbj])
                         )
            else:
                s.append(Subject(id='S' + str(sbj), gender='M', age=0,
                         handedness=''))
        return s

    def _get_paradigm(self):
        '''
        '''
        return ERP(title='ERP_EPFL', stimulation=100,
                   break_duration=300, repetition=20,
                   stimuli=6, phrase='',
                   flashing_mode='SC',
                   speller=[])

    def get_path(self):
        NotImplementedError
