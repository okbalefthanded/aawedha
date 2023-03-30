from aawedha.analysis.preprocess import bandpass, eeg_epoch
from aawedha.paradigms.ssvep import SSVEP
import aawedha.utils.network as network
from aawedha.paradigms.erp import ERP
from aawedha.io.base import DataSet
from scipy.io import loadmat
import numpy as np
import glob
import os


class OpenBMISSVEP(DataSet):
    """
    Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee,
    John Williamson, Siamac Fazli, Seong-Whan Lee, EEG dataset and OpenBMI
    toolbox for three BCI paradigms: an investigation into BCI illiteracy,
    GigaScience, Volume 8, Issue 5, May 2019, giz002,
    https://doi.org/10.1093/gigascience/giz002
    """
    def __init__(self):
        super().__init__(title='OpenBMI_SSVEP',
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
                         doi='https://doi.org/10.1093/gigascience/giz002',
                         url="parrot.genomics.cn")
        self.test_epochs = []
        self.test_y = []
        self.test_events = []
        self.sessions = 100  # index of last trial in a session

    def generate_set(self, load_path=None,
                     download=False,
                     save_folder=None,
                     epoch=[0, 4],
                     band=[4.0, 45.0],
                     order=6, 
                     save=True,
                     fname=None,
                     augment=False,
                     ch=None,
                     downsample=5,
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
        download : bool,
            if True, download raw data first. default False.
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
        if download:
            self.download_raw(load_path)

        if downsample:
            self.fs = self.fs // int(downsample)

        epochs, y, events = self.load_raw(load_path, 'train', epoch,
                                          band, order, ch,
                                          augment, downsample,
                                          method, slide
                                          )
        self.epochs = epochs
        self.y = y
        self.events = events

        epochs, y, events = self.load_raw(load_path, 'test', epoch,
                                          band, order, ch,
                                          augment, downsample,
                                          method, slide
                                          )
        self.test_epochs = epochs
        self.test_y = y
        self.test_events = events

        if ch:
            self.ch_names = [self.ch_names[ch] for ch in self._get_channels(ch)]

        self.subjects = self._get_subjects(n_subjects=54)
        self.paradigm = self._get_paradigm()
        if save:
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
                ev = self._position_to_event(np.array(ev))
                events_subj.append(ev)

            X.append(np.concatenate(x_subj, axis=-1))
            Y.append(np.concatenate(y_subj, axis=-1))
            events.append(np.concatenate(events_subj, axis=-1))

        if augment and mode == 'test':
            self.sessions = self.sessions * augmented

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32).squeeze()
        events = np.array(events).squeeze()
        return X, Y, events

    def download_raw(self, store_path):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        sessions = ['session1', 'session2']
        for sess in sessions:            
            for subj in range(1, 55):
                ftp_client = network.connect_ftp(self.url)
                subj_folder = f"{sess}/s{subj}"
                folder = f"gigadb/pub/10.5524/100001_101000/100542/{subj_folder}"
                path = f"{store_path}/{subj_folder}"
                if not os.path.isdir(path):
                    os.mkdir(path)
                elif any(os.scandir(path)):
                    if network.check_size(ftp_client, path, folder):
                        continue
                network.download_ftp_folder(ftp_client, folder, path, pattern="*SSVEP.mat")
    
    @staticmethod
    def _position_to_event(position):
        """Convert stimulations event in position to its corresponding frequency.
        """
        position_dict = {'down':'5.45', 'right':'6.67', 'left':'8.57', 'up':'12.'}
        idx = np.nonzero(list(position_dict.keys()) == position[:, None])[1]
        events = np.asarray(list(position_dict.values()))[idx]
        return events

    def get_path(self):
        NotImplementedError

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_ON_OFF', control='Sync',
                     stimulation=4000,
                     break_duration=6000, repetition=25,
                     stimuli=4, phrase='',
                     stim_type='ON_OFF',
                     frequencies=['12.', '8.57', '6.67', '5.45']
                     )


class OpenBMIERP(DataSet):
    """
    Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee,
    John Williamson, Siamac Fazli, Seong-Whan Lee, EEG dataset and OpenBMI
    toolbox for three BCI paradigms: an investigation into BCI illiteracy,
    GigaScience, Volume 8, Issue 5, May 2019, giz002,
    https://doi.org/10.1093/gigascience/giz002
    """
    def __init__(self):
        super().__init__(title='OpenBMI_ERP',
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
                         doi='https://doi.org/10.1093/gigascience/giz002',
                         url="parrot.genomics.cn")
        self.test_epochs = []
        self.test_y = []
        self.test_events = []
        self.sessions = 1980  # index of last trial in a session
        self.test_sessions = 2160  # index of last trial in a session

    def generate_set(self, load_path=None,
                     download=False,
                     epoch=[0., .7],
                     band=[1., 10.],
                     order=2, 
                     save=True, 
                     save_folder=None,
                     fname=None,
                     channels=None,
                     downsample=None,
                     ):
        """Main method for creating and saving DataSet objects and files:
            - sets train and test (if present) epochs and labels
            - sets dataset information : subjects, paradigm
            - saves DataSet object as a serialized pickle object

        Parameters
        ----------
        load_path : str
            raw data folder path
        download : bool,
            if True, download raw data first. default False.
        epoch : list
            epoch window start and end in seconds relative to trials' onst
            default : [0, .7]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]
        order : int
            band-pass filter order
            default: 2
        save : bool
            if True save DataSet, default True
        save_folder : str
            DataSet object saving folder path
        fname: str, optional
            saving path for file, specified when different versions of
            DataSet are saved in the same folder
            default: None
        channels : list, optional
            default : None, keep all channels
        downsample: int, optional
            down-sampling factor
            default : None
        """
        if download:
            self.download_raw(load_path)

        if downsample:
            self.fs = self.fs // int(downsample)

        epochs, y, events = self.load_raw(load_path, 'train', epoch,
                                          band, order, channels,
                                          downsample                        
                                          )
        self.epochs = epochs
        self.y = y
        self.events = events

        epochs, y, events = self.load_raw(load_path, 'test', epoch,
                                          band, order, channels,
                                          downsample                                       
                                          )
        self.test_epochs = epochs
        self.test_y = y
        self.test_events = events

        if channels:
            self.ch_names = [self.ch_names[ch] for ch in self._get_channels(channels)]

        self.subjects = self._get_subjects(n_subjects=54)
        self.paradigm = self._get_paradigm()
        if save:
            self.save_set(save_folder, fname)

    def load_raw(self, path=None, mode='', epoch_duration=[0., .7],
                 band=[1., 10.], order=2, ch=None,
                 downsample=None
                 ):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        mode : str
            data acquisition session mode: 'train' or 'test'
        epoch_duration : list
            epoch duration window start and end in seconds relative to trials' onset
            default : [0., .7]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]
        ch : list, optional
            default : None, keep all channels
        order : int
            band-pass filter order
            default: 2
        downsample: int, optional
            down-sampling factor
            default : 4
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
        for subj in range(1, n_subjects+1):
            x_subj, y_subj, events_subj = [], [], []
            for sess in sessions:
                f = glob.glob(f'{path}/{sess}/s{subj}/*ERP.mat')[0]                
                data = loadmat(f)
                data = data['EEG_ERP_'+mode]
                # eeg = data[0][0][0][::stride, :, ch_index].transpose((0, 2, 1))
                # eeg = bandpass(eeg, band, self.fs, order)
                cnt = bandpass(data['x'][0][0][::stride, ch_index], band, self.fs, order)                
                mrk = data['t'][0][0].squeeze() // stride
                eeg = eeg_epoch(cnt, epoch_duration, mrk,
                                self.fs, baseline_correction=True, baseline=0.2)
                y = data['y_dec'][0][0].squeeze().astype(int)
                y[y==2] = 0
                ev = [elm.item() for elm in data['y_class'][0][0].squeeze().tolist()]                
                del data
                del cnt
                x_subj.append(eeg)
                y_subj.append(y)
                # ev = self._position_to_event(np.array(ev))
                events_subj.append(ev)
            X.append(np.concatenate(x_subj, axis=-1))
            Y.append(np.concatenate(y_subj, axis=-1))
            events.append(np.concatenate(events_subj, axis=-1))

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32).squeeze()
        events = np.array(events).squeeze()
        return X, Y, events

    def download_raw(self, store_path):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        sessions = ['session1', 'session2']
        for sess in sessions:            
            for subj in range(1, 55):
                ftp_client = network.connect_ftp(self.url)
                subj_folder = f"{sess}/s{subj}"
                folder = f"gigadb/pub/10.5524/100001_101000/100542/{subj_folder}"
                path = f"{store_path}/{subj_folder}"
                if not os.path.isdir(path):
                    os.mkdir(path)
                elif any(os.scandir(path)):
                    if network.check_size(ftp_client, path, folder):
                        continue
                network.download_ftp_folder(ftp_client, folder, path, pattern="*ERP.mat")

    @staticmethod
    def _position_to_event(position):
        """
        """
        position_dict = {'target':'1', 'nontarget': '2'}
        idx = np.nonzero(list(position_dict.keys()) == position[:, None])[1]
        events = np.asarray(list(position_dict.values()))[idx]
        return events

    def get_path(self):
        NotImplementedError

    def _get_paradigm(self,):
        return ERP(title='ERP_Face', control='Sync',
                   stimulation=80,
                   break_duration=135, repetition=5,
                   stimuli=12,
                   stim_type='Face',
                   speller=['A', 'B', 'C', 'D', 'E', 'F',
                              'G', 'H', 'I', 'J', 'K', 'L',
                              'M', 'N', 'O', 'P', 'Q', 'R',
                              'S', 'T', 'U', 'V', 'W', 'X',
                              'Y', 'Z', '1', '2', '3', '4',
                              '5', '6', '7', '8', '9', '_'],
                   phrase=['NEURAL_NETWORKS_AND_DEEP_LEARNING','PATTERN_RECOGNITION_MACHINE_LEARNING'],
                   flashing_mode='RSP') # random_set_presentation