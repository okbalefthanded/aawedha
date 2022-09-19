from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
from aawedha.utils.network import download_file
from aawedha.utils.utils import unzip_files
from scipy.io import loadmat
import numpy as np
import glob
import re

URLS = ["http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S1-S10.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S11-S20.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S21-S30.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S31-S40.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S41-S50.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S41-S50.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S51-S60.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S61-S70.mat.zip",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/note.txt",
        "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/description.txt"]

class Beta(DataSet):
    """
    Beta Large SSVEP Benchmark
    [1] Liu B, Huang X, Wang Y, Chen X and Gao X (2020) BETA: 
    A Large Benchmark Database Toward SSVEP-BCI Application. 
    Front. Neurosci. 14:627. doi: 10.3389/fnins.2020.00627
    """
    def __init__(self):
        super().__init__(title='Beta_SSVEP',
                         ch_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
                                   'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
                                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1',
                                   'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                                   'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
                                   'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
                                   'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'],

                         fs=250,
                         doi='http://dx.doi.org/10.3389/fnins.2020.00627',
                         url="http://bci.med.tsinghua.edu.cn/upload/liubingchuan"
                         )


    def generate_set(self, load_path=None, download=False, ch=None, epoch=2, band=[5.0, 45.0],
                     order=6, save=True, save_folder=None, fname=None,
                     augment=False, method='divide', slide=0.1):
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
        ch : list, optional
            default : None, keep all channels
        epoch : int
            epoch duration in seconds relative to trials' onset
            default : 2 sec (subjects s1  to s15 full trial's lenght is 2 sec
            3 sec for the rest)
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
        """

        if download:
            self.download_raw(load_path)

        self.epochs, self.y, subj_info = self.load_raw(load_path,ch,
                                            epoch, band, order,
                                            augment, method, slide)
        self.subjects = subj_info
        self.paradigm = self._get_paradigm()
        self.events = self._get_events()
        if save:
            self.save_set(save_folder, fname)

    def load_raw(self, path=None, ch=None, epoch_duration=2,
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
            default : 2 sec (subjects s1  to s15 full trial's lenght is 2 sec
            3 sec for the rest)
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
        n_subjects = 70
        X, Y = [], []
        subj_info = []

        onset = int(0.5 * self.fs)
        # fixed for all subjects, as S16-S70 have
        # 3s stimulation we select the lowest.
        # 2s from S1-S15
        stimulation = 2
        for subj in range(n_subjects):
            data = loadmat(list_of_files[indices == subj][0])
            eeg = data['data']['EEG'][0][0].transpose((1, 0, 3, 2))
            eeg = eeg[:, chans, :, :]
            subj_info.append(self._get_subject(data))
            del data
            eeg = bandpass(eeg, band=band, fs=self.fs, order=order)
            if augment:
                tg = eeg.shape[2]
                v = self._get_augmented_epoched(eeg, ep, stimulation, onset, slide, method)
                eeg = np.concatenate(v, axis=2)
                samples, channels, targets, blocks = eeg.shape
                y = np.tile(np.arange(1, tg + 1), (1, len(v)))
                y = np.tile(y, (1, blocks))
                del v  # saving some RAM
            else:
                eeg = eeg[onset:epoch_duration+onset, :, :, :]
                samples, channels, targets, blocks = eeg.shape
                y = np.tile(np.arange(1, targets + 1), (1, blocks))

            X.append(eeg.reshape((samples, channels, blocks * targets), order='F'))
            Y.append(y)
            del eeg
            del y

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32).squeeze()
        return X, Y, subj_info  

    def download_raw(self, store_path=None):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        for url in URLS:
            download_file(url, store_path)
        # unzip files and delete
        zip_files = glob.glob(f"{store_path}/*.zip")
        unzip_files(zip_files, store_path)        

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
    def _get_subject(data):
        """Extract subject information

        Parameters
        ----------
        data : dict
            subject's raw data

        Returns
        -------
        Subject instance            
        """
        suppl_info = data['data'][0][0][1]
        subj_id = suppl_info['sub'][0][0][0]
        gender = suppl_info['gender'][0][0][0][0]
        age = suppl_info['age'][0][0][0][0]        
        handedness = ''
        condition = 'Healty'
        narrow_snr =  suppl_info['narrow_snr'][0][0][0][0]
        wide_snr = suppl_info['wide_snr'][0][0][0][0]
        bci_quotient = suppl_info['bci_quotient'][0][0][0][0]
         
        return Subject(subj_id, gender, age, handedness,
                       condition, narrow_snr, wide_snr, bci_quotient)

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_JFPM', stimulation=2000, break_duration=500,
                     repetition=6, stimuli=40, phrase='',
                     stim_type='Sinusoidal',
                     frequencies=['8.6',  '8.8',  '9.' ,  '9.2',  '9.4',  '9.6',  '9.8', '10.' , '10.2', 
                                  '10.4', '10.6', '10.8', '11.' , '11.2', '11.4', '11.6', '11.8', '12.' , 
                                   '12.2', '12.4',  '12.6', '12.8', '13.' , '13.2', '13.4', '13.6', '13.8', 
                                   '14.' , '14.2', '14.4', '14.6',  '14.8', '15.' , '15.2', '15.4', '15.6', 
                                   '15.8',  '8.' ,  '8.2',  '8.4'],

                     phase=[4.71238898, 0., 1.57079633,  3.14159265, 4.71238898,
                            0, 1.57079633, 3.14159265, 4.71238898, 0., 1.57079633,
                            3.14159265,  4.71238898, 0., 1.57079633, 3.14159265,
                            4.71238898, 0, 1.57079633, 3.14159265, 4.71238898,
                            0, 1.57079633, 3.14159265, 4.71238898, 0,
                            1.57079633, 3.14159265, 4.71238898, 0, 1.57079633,
                            3.14159265, 4.71238898, 0, 1.57079633, 3.14159265,
                            4.71238898, 0, 1.57079633, 3.14159265
                            ]
                     )

    def get_path(self):
        NotImplementedError
