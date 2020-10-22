from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import glob


class SanDiego(DataSet):
    '''
        San Diego SSVEP joint frequency and phase modulation dataset [1]
        [1] Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
        A Comparison Study of Canonical Correlation Analysis Based Methods for
        Detecting Steady-State Visual Evoked Potentials,"
        PLoS One, vol.10, no.10, e140703, 2015.
    '''

    def __init__(self):
        super().__init__(title='San_Diego',
                         ch_names=['PO7', 'PO3', 'POz',
                                   'PO4', 'PO8', 'O1', 'Oz', 'O2'],
                         fs=256,
                         doi='http://dx.doi.org/10.1371/journal.pone.0140703'
                         )

    def generate_set(self, load_path=None,
                     epoch=1, band=[5.0, 45.0],
                     order=6, save_folder=None,
                     fname=None,
                     augment=False, method='divide',
                     slide=0.1):
        """Main method for creating and saving DataSet objects and files:
            - sets train and test (if present) epochs and labels
            - sets dataset information : subjects, paradigm
            - saves DataSet object as a serialized pickle object

        Parameters
        ----------
        load_path : str
            raw data folder path
        epoch : int
            epoch duration in seconds relative to trials' onset
            default : 1 sec
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [5., 45.]
        order : int
            band-pass filter order
            default: 6
        save_folder : str
            DataSet object saving folder path
        fname: str, optional
            saving path for file, specified when different versions of
            DataSet are saved in the same folder
            default: None
        augment : bool, optional
            if True, EEG data will be epoched following one of
            the data augmentation methods specified by 'method'
            default: False
        method: str, optional
            data augmentation method
            default: 'divide'
        slide : float, optional
            used with 'slide' augmentation method, specifies sliding window
            length.
            default : 0.1

        Returns
        -------
        """
        self.epochs, self.y = self.load_raw(load_path, epoch, band, order, augment, method, slide)
        self.subjects = self._get_subjects(n_subjects=10)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events()
        self.save_set(save_folder, fname)

    def load_raw(self, path=None, epoch_duration=1,
                 band=[5.0, 45.0], order=6, augment=False,
                 method='divide', slide=0.1):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        epoch_duration : int
            epoch duration in seconds relative to trials' onset
            default : 1 sec
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [5., 45.]
        order : int
            band-pass filter order
            default: 6
        augment : bool, optional
            if True, EEG data will be epoched following one of
            the data augmentation methods specified by 'method'
            default: False
        method: str, optional
            data augmentation method
            default: 'divide'
        slide : float, optional
            used with 'slide' augmentation method, specifies sliding window
            length.
            default : 0.1
        Returns
        -------
        x : nd array (subjects x samples x channels x trials)
            epoched EEG data for the entire set or train/test phase
        y : nd array (subjects x trials)
            class labels for the entire set or train/test phase
        """
        list_of_files = sorted(glob.glob(path + 's*.mat'))
        ep = epoch_duration
        epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
        n_subjects = 10
        X, Y = [], []
        stimulation = 4
        onset = 39  # onset in samples
        for subj in range(n_subjects):
            data = loadmat(list_of_files[subj])
            # samples, channels, trials, targets
            eeg = data['eeg'].transpose((2, 1, 3, 0))
            eeg = bandpass(eeg, band=band, fs=self.fs, order=order)
            if augment:
                v = self._get_augmented_epoched(eeg, ep, stimulation, onset, slide, method)
                eeg = np.concatenate(v, axis=2)
                samples, channels, blocks, targets = eeg.shape
                y = np.tile(np.arange(1, targets + 1), (15 * len(v), 1))
                y = y.reshape((1, blocks*targets), order='F')
                del v
            else:
                # epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
                # epoch_duration = np.round(np.array(ep) * self.fs).astype(int)
                eeg = eeg[onset:onset + epoch_duration, :, :, :]
                samples, channels, blocks, targets = eeg.shape
                y = np.tile(np.arange(1, targets + 1), (blocks, 1))
                y = y.reshape((1, blocks * targets), order='F')

            del data  # save some RAM

            X.append(eeg.reshape((samples, channels, blocks * targets), order='F'))
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y

    def _get_events(self):
        """Attaches the experiments paradigm frequencies to
        class labels

        Returns
        -------
        events: nd array (subjects x trials)

        """
        events = np.empty(self.y.shape, dtype=object)
        rows, cols = events.shape
        for i in range(rows):
            for l in range(len(self.paradigm.frequencies)):
                ind = np.where(self.y[i, :] == l+1)
                events[i, ind[0]] = self.paradigm.frequencies[l]

        return events

    @staticmethod
    def _get_paradigm():
        return SSVEP(title='SSVEP_JFPM', stimulation=4000, break_duration=1000,
                     repetition=15, stimuli=12, phrase='', stim_type='ON_OFF',
                     frequencies=['9.25', '11.25', '13.25', '9.75', '11.75', '13.75',
                                  '10.25', '12.25', '14.25', '10.75', '12.75', '14.75'],
                     phase=[0.0, 0.0, 0.0, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi,
                            np.pi, np.pi, np.pi, 1.5 * np.pi, 1.5 * np.pi, 1.5 * np.pi])

    def get_path(self):
        NotImplementedError
