from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass, eeg_epoch
from aawedha.utils.network import download_file
from aawedha.utils.utils import extract_zip
import numpy as np
import pickle
import glob
import gzip
import os


class Exoskeleton(DataSet):
    """
    SSVEP-based BCI of subjects operating an upper limb exoskeleton during a
    shared control

    References :

    [1] Emmanuel K. Kalunga, Sylvain Chevallier, Olivier Rabreau, Eric
    Monacelli. Hybrid interface:
    Integrating BCI in Multimodal Human-Machine Interfaces. IEEE/ASME
    International Conference on Advanced Intelligent Mechatronics (AIM),
    2014, Besancon, France.

    [2] Emmanuel Kalunga, Sylvain Chevallier, Quentin Barthelemy. Data
    augmentation in Riemannian space for Brain-Computer Interfaces,
    STAMLINS (ICML workshop),
    2015, Lille, France.

    [3] Emmanuel K. Kalunga, Sylvain Chevallier, Quentin Barthelemy. Online
    SSVEP-based BCI using Riemannian Geometry. Neurocomputing, 2016.
    arXiv research report on arXiv:1501.03227.

    """
    def __init__(self):
        super().__init__(title='Exoskeleton_SSVEP',
                         ch_names=['Oz', 'O1', 'O2', 'PO3',
                                   'POz', 'PO7', 'PO8', 'PO4'],
                         fs=256,
                         doi='http://dx.doi.org/10.1109/AIM.2014.6878132',
                         url="https://github.com/sylvchev/dataset-ssvep-exoskeleton/archive/refs/heads/master.zip"
                         )
        self.test_epochs = []
        self.test_y = []
        self.test_events = []
        self.alt_url = "https://zenodo.org/records/2392979"

    def generate_set(self, load_path=None,
                     download=False,
                     ch=None,
                     epoch=[2, 5],
                     band=[5., 45.],
                     order=6,
                     save=True,
                     save_folder=None,
                     fname=None,
                     augment=False,
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
        epoch : int
            epoch duration window start and end in seconds relative to trials' onset
            default : [2, 5]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [5., 45.]
        order : int
            band-pass filter order
            default: 6
        save : bool,
            it True save DataSet, default True.
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
        """

        if download:
            self.download_raw(load_path)

        self.epochs, self.y = self.load_raw(load_path, 'train', ch,
                                            epoch, band,
                                            order, augment,
                                            method,
                                            slide
                                            )

        self.test_epochs, self.test_y = self.load_raw(load_path,
                                                      'test', ch, epoch,
                                                      band, order, augment,
                                                      method,
                                                      slide
                                                      )

        self.subjects = self._get_subjects(n_subjects=12)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events(self.y)
        self.test_events = self._get_events(self.test_y)
        if save and save_folder:    
            self.save_set(save_folder, fname)

    def load_raw(self, path=None, mode='', ch=None, epoch_duration=[2, 5],
                 band=[5.0, 45.0], order=6, augment=False,
                 method='divide', slide=0.1):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        mode : str
            data acquisition session mode: 'train' or 'test'
        epoch_duration : int
            epoch duration window start and end in seconds relative to trials' onset
            default : [2, 5]
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
        x : list of nd array (subjects x samples x channels x trials)
            epoched EEG data for the entire set or train/test phase
            trials is not a fixed number, it varies according to subjects sessions
        y : list nd array (subjects x trials)
            class labels for the entire set or train/test phase
            trials is not a fixed number, it varies according to subjects sessions
        """
        files_list = self._get_filelist(path)
        
        if ch:
            chans = self.select_channels(ch)
        else:
            chans = range(len(self.ch_names))
        n_subjects = 12
        X, Y = [], []
        for subj in range(n_subjects):
            session = glob.glob(files_list[subj] + '/*.pz')
            if mode == 'train':
                records = np.arange(0, len(session) - 1).tolist()
            elif mode == 'test':
                records = [len(session) - 1]
            x, y = self._get_epoched(session, records, chans,
                                     epoch_duration, band,
                                     order, augment, method, slide)
            X.append(x.astype(np.float32))
            Y.append(y.astype(np.float32))

        return X, Y

    def download_raw(self, store_path=None):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str,
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        if store_path and not os.path.exists(store_path):
            os.makedirs(store_path)
        fname = f"{store_path}/master.zip"
        download_file(self.url, store_path)        
        extract_zip(fname, store_path)        
        # unzip_files([fname], store_path)

    def to_sync(self):
        """Keep epochs with frequency stimulation and 
        discard idle state trials.
        """
        self.y_orig = self.y[:]
        self.events_orig = self.events  
        if isinstance(self.epochs, list):
            for i, yy in enumerate(self.y):
                    tmp_idx = self.events[i] != 'idle'
                    self.epochs[i] = self.epochs[i][:,:,tmp_idx]
                    self.y[i] = yy[tmp_idx] - 1
        else:
            y_idx = (self.events != 'idle').squeeze()
            self.epochs = self.epochs[:, :, :, y_idx]
            self.y = self.y[:, y_idx] - 1
         
        if hasattr(self, 'test_epochs'):
            self.test_y_orig = self.test_y[:]
            self.testevent_orig = self.test_events[:]            
            if isinstance(self.test_epochs, list):
                for i, yy in enumerate(self.test_y):
                    tmp_idx = self.test_events[i] != 'idle'
                    self.test_epochs[i] = self.test_epochs[i][:,:,tmp_idx]
                    self.test_y[i] = yy[tmp_idx] - 1
            else:
                self.test_y = np.zeros_like(self.test_y)
                self.test_y[self.test_events != 'idle'] = 1.

    def _get_epoched(self, files=[], records=[],
                     ch=None, epoch=[2, 5], band=[5., 45.],
                     order=6, augment=False,
                     method='divide', slide=0.1):
        """Extract epochs from raw continuous EEG file

        Parameters
        ----------
        files : list
            sessions raw data files of a single subject
        records : list
            indices of files to include
        epoch : list
            epoch duration window start and end in seconds relative to trials' onset
            default : [2, 5]
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
        x : nd array (samples x channels x trials)
            epoched EEG data for a single subject in a session mode (train or test)
        y : nd array (subjects x trials)
            class labels for a single subject in a session mode (train or test)
        """
        OVTK_StimulationId_VisualStimulationStart = 32779
        labels = [33024, 33025, 33026, 33027]
        if isinstance(epoch, list):
            epoch = (np.array(epoch) * self.fs).astype(int)
        else:
            epoch = (np.array([0, epoch]) * self.fs).astype(int)
        x, y = [], []
        stimulation = 5 * self.fs
        for sess in records:
            f = gzip.open(files[sess], 'rb')
            df = pickle.load(f, encoding='latin1')
            raw_signal = df['raw_signal'][:, ch] * 1000  # (samples x channels)
            event_pos  = df['event_pos'].reshape((df['event_pos'].shape[0]))
            event_type = df['event_type'].reshape((df['event_type'].shape[0]))
            desc_idx = np.logical_or.reduce([event_type == lb for lb in labels])
            desc = event_type[desc_idx].astype(int)
            # y.append(desc - 33023)
            pos = event_pos[event_type == OVTK_StimulationId_VisualStimulationStart]
            raw_signal = bandpass(raw_signal, band, self.fs, order)
            if augment:
                '''
                stimulation = 5 * self.fs
                augmented = np.floor(stimulation / np.diff(epoch))[0].astype(int)
                v = [eeg_epoch(raw_signal, epoch + np.diff(epoch) * i, pos) for i in range(augmented)]
                '''
                # v = self._get_augmented(raw_signal, epoch, pos, slide=slide, method=method)
                v = self._get_augmented_cnt(raw_signal, epoch, pos, stimulation, slide=slide, method=method)
                epchs = np.concatenate(v, axis=2)
                # y.append(np.tile(desc - 33023, augmented))
                y.append(np.tile(desc - 33023, len(v)))
            else:
                y.append(desc - 33023)
                epchs = eeg_epoch(raw_signal, epoch, pos)
            x.append(epchs)

        if len(x) > 1:
            x = np.concatenate(x, axis=-1)
        else:
            x = np.array(x).squeeze()

        y = np.array(self.flatten(y))

        return x, y  # epochs and labels

    def _get_events(self, y):
        """Attaches the experiments paradigm frequencies to
        class labels
        y : nd array (subjects x trials)
            class labels
        Returns
        -------
        events: list of nd array (subjects x trials)
            stimulation frequencies (or idle) of classes
            trials varies according to subjects sessions
        """
        ev = []
        for i in range(len(y)):
            events = np.empty(y[i].shape, dtype=object)
            for l in range(len(self.paradigm.frequencies)):
                ind = np.where(y[i] == l+1)
                events[ind[0]] = self.paradigm.frequencies[l]
            ev.append(events)
        return ev

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_LED', control='Async',
                     stimulation=5000,
                     break_duration=3000, repetition=8,
                     stimuli=3, phrase='',
                     stim_type='ON_OFF', frequencies=['idle', '13.', '21.', '17.'],
                     )

    def _get_filelist(self, path):
        """Get list of files in a folder

        Parameters
        ----------
        path : str
            folder path

        Returns
        -------
        list of files
        """
        if "dataset-ssvep-exoskeleton-master" in path:
            return sorted(glob.glob(f"{path}/s*"))
        elif os.path.isdir(f"{path}/dataset-ssvep-exoskeleton"):
            return sorted(glob.glob(f"{path}/dataset-ssvep-exoskeleton-master/s*"))

    @staticmethod
    def flatten(list_of_lists=[]):
        """Transforms a list of a list into a single one

        Parameters
        ----------
        list_of_lists : list of lists

        Returns
        -------
        list of items
        """
        return [item for sublist in list_of_lists for item in sublist]

    def get_path(self):
        NotImplementedError