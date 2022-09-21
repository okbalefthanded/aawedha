from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from aawedha.utils.network import download_file
from mne import Epochs, pick_types, events_from_annotations
from mne import set_log_level
from mne.io import concatenate_raws, read_raw_edf
import pandas as pd
import numpy as np
import glob
import os


class Essex(DataSet):
    """P300 Amplitude Dataset

    [1] L. Citi, R. Poli, C. Cinel, Documenting, modelling and exploiting P300 amplitude 
    changes due to variable target delays in Donchin’s speller, J. Neural Eng. 7 (2010). 
    doi:10.1088/1741-2560/7/5/056006.
    """
    def __init__(self):
        super().__init__(title='Essex_P300',
                         ch_names=['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 
                                    'FT7', 'FC5', 'FC3', 'FC1','C1', 'C3', 'C5', 
                                    'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 
                                    'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 
                                    'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 
                                    'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 
                                    'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 
                                    'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 
                                    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
                         fs=2048,
                         doi='http://dx.doi.org/10.1088/1741-2560/7/5/056006',
                         url="https://archive.physionet.org/pn4/erpbci"
                         )
        self.phrase = []
        self.flashes = []

    def generate_set(self, load_path=None, download=False, channels=None, epoch=[0., .7], 
                     band=[1.0, 10.0], order=2,  downsample=None, 
                     save=True, save_folder=None, fname=None,
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
        channels : list, optional
            default : None, keep all channels
        epoch : list
            epoch window start and end in seconds relative to trials' onst
            default : [0, .7]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [1., 10.]
        order : int
            band-pass filter order
            default: 2
        downsample: int, optional
            down-sampling factor
            default : None
        save : bool,
            it True save DataSet, default True.
        save_folder : str
            DataSet object saving folder path
        fname: str, optional
            saving path for file, specified when different versions of
            DataSet are saved in the same folder
            default: None
        """
        if downsample:
            self.fs = self.fs // int(downsample)
        
        if download:
            self.download_raw(load_path)

        self.epochs, self.y, events, phrase = self.load_raw(load_path, channels,
                                                    epoch, band, order, 
                                                    downsample
                                                    )
        if channels:
            self.ch_names = [self.ch_names[ch] for ch in self._get_channels(channels)]

        self.subjects = self._get_subjects()
        self.paradigm = self._get_paradigm()
        self.events = events
        self.phrase = phrase
        if save:
            self.save_set(save_folder, fname)        

    def load_raw(self, path=None, channels=None, epoch=[0., .7], 
                     band=[1.0, 5.0], order=2,  downsample=None):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        channels : list of str
            list of channels to keep, default None, select all channels.
        epoch : list
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
        X : list of nd array (samples x channels x trials_per_subject)
            epoched EEG data for the entire set or train/test phase
        Y : list of 1d array (trials_per_subject)
            class labels for the entire set or train/test phase
        ev : list of 1d array (trials_per_subject)
            frequency stimulation of each class
        """
        set_log_level(verbose=False)
        subjects = range(1, 13)
        X, Y , ev, phrase = [], [], [], []
        flashes = []
        ev_id = {'AGMSY5': 1, 'BHNTZ6': 2, 'CIOU17': 3, 'DJPV28': 4,
                 'EKQW39': 5, 'FLRX4_': 6, 'ABCDEF': 7, 'GHIJKL': 8,
                 'MNOPQR': 9, 'STUVWX': 10, 'YZ1234':11, '56789_': 12
                 }
        
        for subj in subjects:
            if subj < 10:
                subj = '0'+str(subj)
            raw_names = sorted(glob.glob(f"{path}/s{subj}/*.edf"))
            raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_names])
            s = pd.Series(raw.annotations.description)            
            targets = s[s.str.startswith("#Tgt")]           
            targets = [tar[4] for tar in targets]   
            # FROM SUBJECT 3, ANNOTATIONS DON'T CONTAIN #start
            # THE DATE OF RECORDING REPLACES #start
            # AND CONTAIN A NEW count: #counted20of20         
            #starts = s[s.str.startswith("#start")]            
            # targets = s[s.str.startswith("#Tgt")].str.split("_")
            # targets = [tar[0][-1] for tar in targets]            
            # starts = s[s.str.startswith("#start")]
            #if starts.size == 0:
            starts = s[s.str.startswith("#Tgt")]
            ends = s[s.str.startswith("#end")]
            counts = s[s.str.startswith("#counted")]
            if counts.empty:
                start_index = np.where("#start" == raw.annotations.description)
                end_index = np.where("#end" == raw.annotations.description)
                diff = (end_index[0]- start_index[0] -1) / 6
                counts = diff.tolist()
            else:
                counts = [int(c[12:14]) for c in counts]           
            desc = list(np.unique(raw.annotations.description))
            desc = [dc for dc in desc if len(dc)==6 and dc != '#start']
            event_keys = list(ev_id.keys())
            if sorted(desc) != sorted(event_keys):
                # subjects : 6 to 12, character '_' is replace by '#'
                old_keys = sorted(list(set(event_keys).difference(desc))) 
                new_keys = sorted(list(set(desc).difference(event_keys)))
                evs = ev_id.copy()
                for idx, key in enumerate(new_keys):
                    evs[key] = evs.pop(old_keys[idx])
                events, _ = events_from_annotations(raw, event_id=evs) 
            else:
                # subjects: 1 to 5
                events, _ = events_from_annotations(raw, event_id=ev_id)            
            r = [ev_id[key] if key in ev_id else key for key in raw.annotations.description]
            ss = pd.Series(r)
            y = self._get_labels(targets, ss, starts, ends, ev_id)
            if channels:
                eeg_channels = channels
            else:
                eeg_channels = raw.info.ch_names[:64]
                
            raw.pick_channels(eeg_channels)
            raw.filter(band[0], band[1], method='iir', iir_params=dict(order=order, ftype='butter'))
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')

            # epos = Epochs(raw, events, ev_id, epoch[0], epoch[1], proj=True, picks=picks,
            #               baseline=(-0.2, 0), preload=True)
            epos = Epochs(raw, events, ev_id, -0.2, epoch[1], proj=True, picks=picks,
                           baseline=(-0.2, 0), preload=True)
            epos.crop(epoch[0], epoch[1])
            if downsample:
                epos.decimate(downsample)
            epos = np.transpose(epos.get_data().astype(np.float32), (2, 1, 0))
            X.append(epos*1e6)
            Y.append(y.astype(np.float32))
            ev.append(events[:, -1])
            phrase.append(targets)
            flashes.append(counts)
            del raw, epos, y, events, targets, counts, s, ss, picks
        
        #X = np.array(X)
        #Y = np.array(Y).squeeze()
        #ev = np.array(ev).squeeze()
        self.flashes = flashes
        return X, Y, ev, phrase

    def download_raw(self, store_path=None):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        record_file = f"{self.url}/RECORDS"
        download_file(record_file, store_path)
        urls = self._get_files_urls(store_path)
        for i in range(12):
            # mkdir for subject
            if i < 10:
                subj_folder = f"{store_path}/s0{i+1}"
            else:
                subj_folder = f"{store_path}/s{i+1}"
            if not os.path.isdir(subj_folder):
                os.mkdir(subj_folder)
            # download edf and edf.event file for subject
            for subj_urls in urls[i]:
                download_file(subj_urls, subj_folder)

    @staticmethod
    def _get_labels(targets, events, starts, ends, ev_id):
        """Extract labels from description.
        Returns
        -------
        1d array
            trials labels 0/1 : 0 non target, 1 target
        """
        y = []
        for idx, tr in enumerate(targets):                    
            '''
            if events[ends.index[idx]-1] != '#end':
                trial_events = events[starts.index[idx]+2:ends.index[idx]-1]
            else:
                trial_events = events[starts.index[idx]+2:ends.index[idx]]
            '''
            if type(events[ends.index[idx]-1]) is int:
                trial_events = events[starts.index[idx]+2:ends.index[idx]]
            else:
                trial_events = events[starts.index[idx]+2:ends.index[idx]-1]
            trial_target = [ev_id[key]for key in ev_id if tr in key]
            y_trial = np.zeros(len(trial_events))
            target_indices = np.logical_or(trial_events==trial_target[0], trial_events==trial_target[1])
            y_trial[target_indices] = 1.
            y.append(y_trial)
        return np.hstack(y)
    
    def _get_paradigm(self):
        return ERP(title='Essex_P300', stimulation=100,
                   break_duration=50, repetition=20,
                   stimuli=12, phrase='',
                   flashing_mode='RC',
                   speller=[])
    
    def get_path(self):
        NotImplementedError

    def _get_files_urls(self, store_path):
        """Return each subject files url by reading RECORDS file.

        Parameters
        ----------
        store_path : str
            folder path where record file is stored

        Returns
        -------
        list of lists
            each list contains subject file urls
        """
        with open(f"{store_path}/RECORDS", "r") as f:
            recs = f.read()
        files = recs.split('\n')
        urls = []
        for i in range(1,13):
            tmp = []
            for f in files:
                if i < 10:
                    ptrn = f"s0{i}"
                else:
                    ptrn = f"s{i}"
                if f.startswith(ptrn):
                    tmp.extend([f"{self.url}/{f}", f"{self.url}/{f}.event"])                
            urls.append(tmp)
        return urls
