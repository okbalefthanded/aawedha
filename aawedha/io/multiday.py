from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass, eeg_epoch
import numpy as np
import glob
import h5py


class MultiDay(DataSet):
    """
    Ga-Young Choi, Chang-Hee Han, Young-Jin Jung, Han-Jeong Hwang,
    A multi-day and multi-band dataset for a steady-state visual-evoked
    potential–based brain-computer interface, GigaScience, Volume 8,
    Issue 11, November 2019, giz133, https://doi.org/10.1093/gigascience/giz133
    """

    def __init__(self):
        super().__init__(title='Multiday',
                         ch_names=['Fp1', 'Fp2', 'AF4', 'AF3', 'F5',
                                   'Fz', 'FC1', 'FC5', 'F6', 'FC2',
                                   'FC6', 'C4', 'Cz', 'C3', 'CP1',
                                   'CP2', 'CP6', 'P8', 'P4', 'Pz',
                                   'POz', 'PO4', 'PO8', 'O2', 'Oz',
                                   'O1', 'PO3', 'P3', 'CP5', 'P7',
                                   'PO7', 'T7', 'T8'],
                         fs=200,
                         doi='https://doi.org/10.1093/gigascience/giz133')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, mode='', epoch_duration=[0, 6],
                 band=[3.0, 50.0], order=6, augment=False):
        '''
        '''
        if isinstance(epoch_duration, list):
            epoch_duration = (np.array(epoch_duration) * self.fs).astype(int)
        else:
            epoch_duration = (
                np.array([0, epoch_duration]) * self.fs).astype(int)

        if mode == 'train':
            session = '/Day1'
        else:
            session = '/Day2'

        subjects_list = glob.glob(path+'/S*')
        subjects_list.pop(23)  # exclude S24 because of a faulty file
        X, Y = [], []
        records = 6

        for subj in subjects_list:
            k = 0
            x_subj, y_subj = [], []
            cnt_list = glob.glob(subj+session+'/cnt*')
            mrk_list = glob.glob(subj+session+'/mrk*')
            for i in range(records):
                cnt = h5py.File(cnt_list[i], 'r')
                mrk = h5py.File(mrk_list[i], 'r')
                # raw continuous EEG (samples x channels)
                x = cnt['cnt/x'].value.T
                y_orig = mrk['mrk/event/desc'].value.astype(int).squeeze()
                markers_orig = np.around(
                    mrk['mrk/time'].value / 5).astype(int).squeeze()
                #
                x = x[:, :33]  # keeps only EEG channels
                x = bandpass(x, band, self.fs, order)
                only_stimulations = y_orig != 5
                if i >= 2 and i < 4:
                    k = 4
                elif i >= 4:
                    k = 8
                y = y_orig[only_stimulations] + k
                markers = markers_orig[only_stimulations]
                if augment:
                    stimulation = 6 * self.fs
                    augmented = np.floor(
                        stimulation / np.diff(epoch_duration))[0].astype(int)
                    v = [eeg_epoch(x, epoch_duration + np.diff(epoch_duration)
                                   * i, markers) for i in range(augmented)]
                    eeg = np.concatenate(v, axis=2)
                    y = np.tile(y, augmented)
                else:
                    eeg = eeg_epoch(x, epoch_duration, markers)
                #
                cnt.close()
                mrk.close()
                x_subj.append(eeg)
                y_subj.append(y)

            X.append(np.concatenate(x_subj, axis=-1))
            Y.append(np.concatenate(y_subj, axis=-1))

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y

    def generate_set(self, load_path=None, epoch=[0, 6],
                     band=[3.0, 50.0],
                     order=6, save_folder=None,
                     augment=False):
        '''
        '''
        self.epochs, self.y = self.load_raw(load_path, 'train',
                                            epoch, band,
                                            order, augment
                                            )

        self.test_epochs, self.test_y = self.load_raw(load_path,
                                                      'test', epoch,
                                                      band, order, augment
                                                      )
        self.subjects = self._get_subjects(n_subjects=29)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events(self.y)
        self.test_events = self._get_events(self.test_y)
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
                     stimulation=6000,
                     break_duration=6000, repetition=20,
                     stimuli=12, phrase='',
                     stim_type='LED',
                     frequencies=['40.0', '40.5', '41.0', '41.5',
                                  '5.0', '5.5', '6.0', '6.5',
                                  '21.0', '21.5', '22.0', '22.5',
                                  ]
                     )

    def h5ref_to_strings(self, hf, ref):
        """convert HDF5 reference object to String.

        Parameters
        ----------
        hf : HDF5 file
            an instance of an opened HDF5 file containing
            the group/dataset
        ref : HDF5 Dataset
            HDF5 dataset of strings

        Returns
        -------
        List of strings
        """
        shape = ref.value.shape
        if shape[1] > shape[0]:
            array = ref.value.T
        else:
            array = ref.value
        return [''.join(chr(i) for i in hf[obj[0]]) for obj in array]
