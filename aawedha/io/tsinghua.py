from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import re
import glob


class Tsinghua(DataSet):
    """
    Tsinghua SSVEP sampled sinusoidal joint frequency-phase modulation (JFPM)
    [1} X. Chen, Y. Wang, M. Nakanishi, X. Gao, T. -P. Jung, S. Gao,
       "High-speed spelling with a noninvasive brain-computer interface",
       Proc. Int. Natl. Acad. Sci. U. S. A, 112(44): E6058-6067, 2015.
    """

    def __init__(self):
        super().__init__(title='Tsinghua',
                         ch_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
                                   'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
                                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1',
                                   'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                                   'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
                                   'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
                                   'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'],
                         fs=250,
                         doi='http://dx.doi.org/10.1073/pnas.1508080112'
                         )

    def generate_set(self, load_path=None, ch=None, epoch=1, band=[5.0, 45.0],
                     order=6, save_folder=None, fname=None,
                     augment=False, method='divide', slide=0.1):
        """Main method for creating and saving DataSet objects and files:
            - sets train and test (if present) epochs and labels
            - sets dataset information : subjects, paradigm
            - saves DataSet object as a serialized pickle object

        Parameters
        ----------
        load_path : str
            raw data folder path
        ch : list, optional
            default : None, keep all channels
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
        self.epochs, self.y = self.load_raw(load_path,ch,
                                            epoch, band, order,
                                            augment, method, slide)
        self.subjects = self._get_subjects(path=load_path)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events()
        self.save_set(save_folder, fname)

    def load_raw(self, path=None, ch=None, epoch_duration=1,
                 band=[5.0, 45.0], order=6, augment=False,
                 method='divide', slide=0.1):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        ch : list, optional
            subset of channels to keep in DataSet from the total montage.
            default: None, keep all channels
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
        y : nd array (subjects x n_classes)
            class labels for the entire set or train/test phase
        """
        if ch:
            chans = self.select_channels(ch)
        else:
            chans = range(len(self.ch_names))
        list_of_files = np.array(glob.glob(path + '/S*.mat'))
        indices = np.array([int(re.findall(r'\d+', n)[0]) for n in list_of_files]) - 1
        ep = epoch_duration
        epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
        n_subjects = 35
        X, Y = [], []
        # augmented = 0
        onset = int(0.5 * self.fs)
        stimulation = 5
        for subj in range(n_subjects):
            data = loadmat(list_of_files[indices == subj][0])
            eeg = data['data'].transpose((1, 0, 2, 3))
            eeg = eeg[:, chans, :, :]
            del data
            eeg = bandpass(eeg, band=band, fs=self.fs, order=order)
            if augment:
                tg = eeg.shape[2]
                v = self._get_augmented_epoched(eeg, ep, stimulation, onset, slide, method)
                eeg = np.concatenate(v, axis=2)
                samples, channels, targets, blocks = eeg.shape
                # y = np.tile(np.arange(1, tg + 1), (1, augmented))
                y = np.tile(np.arange(1, tg + 1), (1, len(v)))
                y = np.tile(y, (1, blocks))
                del v  # saving some RAM
            else:
                # epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
                eeg = eeg[onset:epoch_duration, :, :, :]
                samples, channels, targets, blocks = eeg.shape
                y = np.tile(np.arange(1, targets + 1), (1, blocks))

            X.append(eeg.reshape((samples, channels,blocks * targets), order='F'))
            Y.append(y)
            del eeg
            del y

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
    def _get_subjects(n_subject=0, path=None):
        """Read dataset subject info file and retrieve information

        Parameters
        ----------
        n_subject : int
            number of subjects in dataset
        path : str
          raw data folder path

        Returns
        -------
        list of Subject instance
        """
        sub_file = path + '/Sub_info.txt'
        with open(sub_file, 'r') as f:
            info = f.read().split('\n')[2:]
        return [Subject(id=s.split()[0], gender=s.split()[1],
                        age=s.split()[2], handedness=s.split()[3])
                for s in info if len(s) > 0]

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_JFPM', stimulation=5000, break_duration=500,
                     repetition=6, stimuli=40, phrase='',
                     stim_type='Sinusoidal',
                     frequencies=['8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.', '8.2', '9.2',
                                  '10.2', '11.2', '12.2', '13.2', '14.2', '15.2', '8.4', '9.4', '10.4', '11.4',
                                  '12.4', '13.4', '14.4', '15.4', '8.6', '9.6', '10.6', '11.6', '12.6', '13.6',
                                  '14.6', '15.6', '8.8', '9.8', '10.8', '11.8', '12.8', '13.8', '14.8', '15.8'],

                     phase=[0., 1.57079633, 3.14159265, 4.71238898, 0.,
                            1.57079633, 3.14159265, 4.71238898, 1.57079633, 3.14159265,
                            4.71238898, 0., 1.57079633, 3.14159265, 4.71238898,
                            0., 3.14159265, 4.71238898, 0., 1.57079633,
                            3.14159265, 4.71238898, 0., 1.57079633, 4.71238898,
                            0., 1.57079633, 3.14159265, 4.71238898, 0.,
                            1.57079633, 3.14159265, 0., 1.57079633, 3.14159265,
                            4.71238898, 0., 1.57079633, 3.14159265, 4.71238898]
                     )

    def get_path(self):
        NotImplementedError
