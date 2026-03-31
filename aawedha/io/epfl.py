from aawedha.analysis.preprocess import bandpass, eeg_epoch
from aawedha.analysis.utils import array_to_intstr
from aawedha.utils.network import download_file
from aawedha.paradigms.subject import Subject
from aawedha.utils.utils import extract_zip
from aawedha.paradigms.erp import ERP
from aawedha.io.base import DataSet
from scipy.io import loadmat
from datetime import datetime
import numpy as np
import glob
import os

class EPFL(DataSet):
    """
        EPFL Image speller Dataset [1]

        Reference:
        Ulrich Hoffmann, Jean-Marc Vesin, Karin Diserens, and Touradj Ebrahimi.
        An efficient P300-based brain-compuer interface for disabled subjects.
        Journal of Neuroscience Methods, 2007

    """

    def __init__(self):
        super().__init__(title='EPFL_Image_Speller',
                         ch_names=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7',
                                   'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3',
                                   'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6',
                                   'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8',
                                   'AF4', 'Fp2', 'Fz', 'Cz'],
                         fs=512,
                         doi='https://doi.org/10.1016/j.jneumeth.2007.03.005',
                         url='https://documents.epfl.ch/groups/m/mm/mmspg/www/BCI/p300')
                          
        self.test_epochs = []
        self.test_y = []
        self.test_events = []
        self.phrase = []
        self.test_phrase = []
        self.flashes = []
        self.test_flashes = []

    def generate_set(self, 
                     load_path=None,
                     download=False,
                     save=True,
                     save_folder=None,
                     fname=None,
                     epoch=[0., 0.7],
                     band=[1, 10],
                     order=2,
                     baseline=0.2,
                     channels=None,
                     downsample=None,
                     train_repetition=None,
                     test_repetition=None):
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
        
        save_folder : str
            DataSet object saving folder path

        epoch : list of float
            epoch duration in seconds relative to trials' onset
            default : 700 msec, [0., .7]
        
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]
        
        order : int
            band-pass filter order
            default: 2
        """
        if download:
            self.download_raw(load_path)
        self.epochs, self.y, self.test_epochs, self.test_y = self.load_raw(
            load_path, epoch, band, order, baseline)
        
        self.subjects = self._get_subjects(n_subjects=9)
        self.paradigm = self._get_paradigm()

        if train_repetition:
            self.epochs, self.y, self.events = self._select_test_trials("train", train_repetition)
            self.paradigm.repetition = train_repetition
        if test_repetition:
            self.test_epochs, self.test_y, self.test_events = self._select_test_trials("test", test_repetition)
            self.paradigm.online_repetition = test_repetition

        if save:
            self.save_set(save_folder, fname)
        return self

    def load_raw(self, path=None, epoch=[0., .7],
                 band=[1, 10], order=2, baseline=0.3):
        """Read and process raw data into structured arrays
        Note: subject 5 is excluded from the dataset.
        Parameters
        ----------
        path : str
            raw data folder path
        
        epoch : list of float
            epoch duration in seconds relative to trials' onset
            default : 700 msec [0., 0.7]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]
        order : int
            band-pass filter order
            default: 2

        Returns
        -------
        X : nd array (subjects x samples x channels x trials)
            epoched EEG data for the train
        Y : nd array (subjects x n_classes)
            class labels for the train
        X_test : nd array (subjects x samples x channels x trials)
            epoched EEG data for the test phase
        Y_test : nd array (subjects x n_classes)
            class labels for the test phase        
        """
        subjects = 9
        sessions = range(1, 5)
        events = []
        events_test = []

        X, Y = [], []
        X_test, Y_test = [], []
        ph, ph_test    = [], []
        calib_flashes, test_flashes = [], []
        for sbj in range(1, subjects + 1):
            target = []
            if sbj == 5:
                continue
            raw_names = ['{f}/subject{s}/session{r}/'.format(f=path, s=sbj, r=r) for r in sessions]
            epochs = []
            y      = []
            stims  = []
            flsh   = []
            # train sessions
            for session in range(len(raw_names)-1):
                files = glob.glob(raw_names[session] + '*.mat')
                ep, yy, trg, stm, cb_f = self._load_session(files, epoch, band, order, baseline)  # subject, session, runs
                epochs.append(ep)
                y.append(yy)
                target.append(trg)
                stims.append(stm)
                flsh.append(cb_f)
            # test session
            test_files = glob.glob(raw_names[-1] + '*.mat')
            test_epochs, test_y, test_target, stm, flashes = self._load_session(test_files, epoch, band, order, baseline)

            epochs = np.concatenate(epochs, axis=-1)  # subject, sessions, runs
            y      = np.concatenate(y, axis=-1)
            flsh   = np.concatenate(flsh)
            ph.append(array_to_intstr(np.concatenate(target, axis=-1)))
            # ph.append(target)
            # ph_test.append(np.concatenate(test_target, axis=-1))
            ph_test.append(array_to_intstr(test_target))
            X.append(epochs.astype(np.float32))
            Y.append(y.astype(np.float32))
            X_test.append(test_epochs.astype(np.float32))
            Y_test.append(test_y.astype(np.float32))
            events.append(np.concatenate(stims, axis=-1))
            events_test.append(stm)
            calib_flashes.append(flsh)
            test_flashes.append(flashes)
            
        #
        self.events = events
        self.test_events = events_test
        self.flashes = np.array(calib_flashes)
        self.phrase = np.array(ph)
        # self.phrase_test = np.array(ph_test)
        self.test_phrase  = np.array(ph_test)
        self.test_flashes = np.array(test_flashes)
        return X, Y, X_test, Y_test

    def download_raw(self, store_path=None):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        urls = self.get_urls()
        for url in urls:
            download_file(url, store_path)
        # unzip files and delete
        zip_files = glob.glob(f"{store_path}/*.zip")
        for zipf in zip_files:
            extract_zip(zipf, store_path)
            os.remove(zipf)
        # unzip_files(zip_files, store_path)

    def get_urls(self):
        """Get dataset files urls.

        Returns
        -------
        list of str
            dataset files urls.
        """
        urls = []
        for i in range(1, 10):
            url_f = f"{self.url}/subject{i}.zip"
            urls.append(url_f)
        return urls 

    def _load_session(self, files, epoch, band, order, baseline):
        """Process a single session files (multiple runs for each subject): load, filter, epoch.

        Parameters
        ----------
        files : list
            session runs files paths.

        epoch : list of float
            epoch duration in seconds relative to trials' onset
            default : 700 msec, [0., .7]

        band : list of int
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]

        order : int
            band-pass filter order
            default: 2

        Returns
        -------
        epochs : ndarray (samples x channels x trials)
            EEG epochs
        y : 1d array (trials)
            epochs labels 0/1 : 0 non target, 1 target
        target : int
            target label value
        stims : 1d array (trials)
            trials stimulus
        """
        epochs = []
        y = []
        target = []
        stims = []
        flashes = []
        for run in range(len(files)):
            data = loadmat(files[run])
            ep, y_tmp, trg, stim = self._get_epochs(data, epoch, band, order, baseline)
            epochs.append(ep)
            y.append(y_tmp)
            target.append(trg)
            stims.append(stim)
            flashes.append(len(stim))

        # epochs = np.array(epochs)
        epochs = np.concatenate(epochs, axis=-1)
        # y = np.array(y)
        y = np.concatenate(y, axis=-1)
        target = np.array(target)
        stims = np.concatenate(stims, axis=-1)
        return epochs, y, target, stims, flashes

    def _get_epochs(self, data, epoch, band, order, baseline):
        """Process a single run file for a subject.

        Parameters
        ----------
        data : HDF5 instance
            .mat file containg data

        epoch : list of float
            epoch duration in seconds relative to trials' onset
            default : 700 msec, [0., .7]

        band : list of int
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]

        order : int
            band-pass filter order
            default: 2

        Returns
        -------
        epochs : ndarray (samples x channels x trials)
            EEG epochs
        y : 1d array (trials)
            epochs labels 0/1 : 0 non target, 1 target
        target : int
            target label value
        stims : 1d array (trials)
            trials stimulus
        """
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
        epochs = eeg_epoch(signal, epoch_length, pos, 
                            self.fs, baseline_correction=True, baseline=baseline)
        y = np.zeros(n_trials)
        y[stimuli == target] = 1

        return epochs, y, target, stimuli
    
    def _select_test_trials(self, mode, repetition):
        """Select a fixed number of trials prediction
        scores for EPFL P300 dataset

        Parameters
        ----------    
        mode : str
            "train" or "test", to select from train or test sessions
        repetition : int
            count of trials repetition
        """
        flash_attr ={
            "train": "flashes",
            "test": "test_flashes"
        }
        x_attr = {
                "train": "epochs",
                "test": "test_epochs"
        }
        y_attr = {
            "train": "y",
            "test": "test_y"
        }

        ev_attr = {
            "train": "events",
            "test": "test_events"
        }
        n_subj = len(self.epochs)
        # n_char = 18 if mode == "train" else 6 
        trials = getattr(self, flash_attr[mode]) // self.paradigm.stimuli
        max_steps = repetition * self.paradigm.stimuli        
        epochs, labels, events = [], [], []
        for i in range(n_subj):
            x, y, ev = [], [], [] 
            step  = 0
            k     = 0
            for tr in np.nditer(trials[i]):
                step = (self.paradigm.stimuli * tr) + k
                args = np.arange(k, step)
                # x.append(self.test_epochs[i][:,:,args][:,:,:max_steps])
                # y.append(self.test_y[i].squeeze()[args][:max_steps])
                # ev.append(self.test_events[i].squeeze()[args][:max_steps])
                x.append( getattr(self, x_attr[mode])[i][:,:,args][:,:,:max_steps] )
                y.append( getattr(self, y_attr[mode])[i].squeeze()[args][:max_steps] )
                ev.append( getattr(self, ev_attr[mode])[i].squeeze()[args][:max_steps] )
                k = step
            x   = np.concatenate(x, axis=-1)
            y   = np.concatenate(y).squeeze()    
            ev  = np.concatenate(ev).squeeze()
            epochs.append(x)
            labels.append(y)
            events.append(ev)
        return epochs, labels, events

    def _get_subjects(self, n_subjects=0):
        """Construct Subjects info list from subjects info files.

        Parameters
        ----------
        n_subjects : int, optional
            sujbects count in dataset, by default 0

        Returns
        -------
        list
            of Subject objects containing subjects infos.
        """
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
        """Creates paradigm object.
        
        Returns 
        -------
        Paradigm instance.
        """
        return ERP(title='ERP_EPFL', stimulation=100,
                   break_duration=300, repetition=20,
                   stimuli=6, phrase='',
                   flashing_mode='SC',
                   speller=['1','2', '3','4','5','6'])

    def get_path(self):
        NotImplementedError
