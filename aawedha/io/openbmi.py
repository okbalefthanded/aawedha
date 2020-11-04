from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass, eeg_epoch
from scipy.io import loadmat
import numpy as np
import glob


class OpenBMISSVEP(DataSet):
    """
    Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee,
    John Williamson, Siamac Fazli, Seong-Whan Lee, EEG dataset and OpenBMI
    toolbox for three BCI paradigms: an investigation into BCI illiteracy,
    GigaScience, Volume 8, Issue 5, May 2019, giz002,
    https://doi.org/10.1093/gigascience/giz002
    """

    def __init__(self):
        super().__init__(title='OpenBMI SSVEP',
                         ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz',
                                   'F4', 'F8', 'FC5', 'FC1', 'FC2',
                                   'FC6', 'T7', 'C3', 'Cz', 'C4',
                                   'T8', 'TP9', 'CP5', 'CP1', 'CP2',
                                   'CP6', 'TP10', 'P7', 'P3', 'Pz',
                                   'P4', 'P8', 'PO9', 'O1', 'Oz',
                                   'O2', 'PO10', 'FC3', 'FC4', 'C5',
                                   'C1', 'C2', 'C6', 'CP3', 'CPz',
                                   'CP4', 'P1', 'P2', 'POz', 'FT9',
                                   'FTT9h', 'TTP7h', 'TP7', 'TPP9h',
                                   'FT10', 'FTT10h', 'TPP8h', 'TP8',
                                   'TPP10h', 'F9', 'F10', 'AF7',
                                   'AF3', 'AF4', 'AF8', 'PO3',
                                   'PO4'],
                         fs=1000,
                         doi='https://doi.org/10.1093/gigascience/giz002')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []
        self.sessions = 100  # index of last trial in a session

    def generate_set(self, load_path=None,
                     epoch=[0, 4],
                     band=[4.0, 45.0],
                     order=6, save_folder=None,
                     fname=None,
                     augment=False,
                     channels=None,
                     downsample=None,
                     method='divide',
                     slide=0.1):
        """Main method for creating and saving DataSet objects and files:
            - sets train and test (if present) epochs and labels
            - sets dataset information : subjects, paradigm
            - saves DataSet object as a serialized pickle object

        Parameters
        ----------
        load_path : str
            raw data folder path
        epoch : list
            epoch window start and end in seconds relative to trials' onst
            default : [0, 4]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [4., 45.]
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
        channels : list, optional
            default : None, keep all channels
        downsample: int, optional
            down-sampling factor
            default : None
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
        if downsample:
            self.fs = self.fs // int(downsample)

        epochs, y, events = self.load_raw(load_path, 'train', epoch,
                                          band, order, channels,
                                          augment, downsample,
                                          method, slide
                                          )
        self.epochs = epochs
        self.y = y
        self.events = events

        epochs, y, events = self.load_raw(load_path, 'test', epoch,
                                          band, order, channels,
                                          augment, downsample,
                                          method, slide
                                          )
        self.test_epochs = epochs
        self.test_y = y
        self.events = events

        if channels:
            self.ch_names = [self.ch_names[ch] for ch in self._get_channels(channels)]

        self.subjects = self._get_subjects(n_subjects=54)
        self.paradigm = self._get_paradigm()
        self.save_set(save_folder,fname)

    def load_raw(self, path=None, mode='', epoch_duration=[0, 4],
                 band=[4.0, 45.0], order=6, ch=None,
                 augment=False, downsample=None,
                 method='divide', slide=0.1):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        mode : str
            data acquisition session mode: 'train' or 'test'
        epoch_duration : list
            epoch duration window start and end in seconds relative to trials' onset
            default : [0, 4]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [5., 45.]
        ch : list, optional
            default : None, keep all channels
        order : int
            band-pass filter order
            default: 6
        augment : bool, optional
            if True, EEG data will be epoched following one of
            the data augmentation methods specified by 'method'
            default: False
        downsample: int, optional
            down-sampling factor
            default : 4
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
        events : nd array (subjects x n_classes)
            frequency stimulation of each class
        """
        ch_index = self._get_channels(self.ch_names)
        if ch:
            ch_index = self._get_channels(ch)

        stride = 1
        if downsample:
            stride = int(downsample)

        sessions = ['session1', 'session2']
        n_subjects = 54
        if isinstance(epoch_duration, list):
            epoch_duration = (np.array(epoch_duration) * self.fs).astype(int)
        else:
            epoch_duration = (np.array([0, epoch_duration]) * self.fs).astype(int)
        X, Y = [], []
        events = []
        stimulation = 4 * self.fs
        for subj in range(1, n_subjects+1):
            x_subj, y_subj, events_subj = [], [], []
            for sess in sessions:
                f = glob.glob(f'{path}/{sess}/s{subj}/*SSVEP.mat')[0]
                data = loadmat(f)
                data = data['EEG_SSVEP_'+mode]
                cnt = bandpass(data[0][0][1][::stride, ch_index], band, self.fs, order)
                mrk = data[0][0][2].squeeze() // stride
                y = data[0][0][4].squeeze().astype(int)
                ev = [elm.item() for elm in data[0][0][6].squeeze().tolist()]
                if augment:
                    v = self._get_augmented_cnt(cnt, epoch_duration, mrk, stimulation, slide, method)
                    augmented = len(v)
                    eeg = np.concatenate(v, axis=2)
                    y = np.tile(y, augmented)
                    ev = np.tile(ev, augmented)
                    del v
                else:
                    augmented = 1
                    eeg = eeg_epoch(cnt, epoch_duration, mrk)
                del data
                del cnt
                x_subj.append(eeg)
                y_subj.append(y)
                events_subj.append(ev)

            X.append(np.concatenate(x_subj, axis=-1))
            Y.append(np.concatenate(y_subj, axis=-1))
            events.append(np.concatenate(events_subj, axis=-1))

        if augment and mode == 'test':
            self.sessions = self.sessions * augmented

        X = np.array(X)
        Y = np.array(Y).squeeze()
        events = np.array(events).squeeze()
        return X, Y, events



    def get_path(self):
        NotImplementedError

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_ON_OFF', control='Sync',
                     stimulation=4000,
                     break_duration=6000, repetition=25,
                     stimuli=4, phrase='',
                     stim_type='ON_OFF',
                     frequencies=['12', '8.57', '6.67', '5.45']
                     )
