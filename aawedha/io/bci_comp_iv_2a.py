from aawedha.analysis.preprocess import bandpass, eeg_epoch
from aawedha.paradigms.motor_imagery import MotorImagery
from aawedha.paradigms.subject import Subject
from aawedha.utils.network import download_file
from aawedha.io.base import DataSet
from mne import set_log_level
from scipy.io import loadmat
import numpy as np
import glob


from mne.io.edf.edf import read_raw_edf


class Comp_IV_2a(DataSet):
    """
        Motor Imagery dataset [1]

        Reference :
        [1]  M. Tangermann, K.-R. Muller, A. Aertsen, N. Birbaumer, C. Braun, C. Brunner, R. Leeb, C. Mehring,
        K. Miller, G. Mueller-Putz, G. Nolte, G. Pfurtscheller, H. Preissl, G. Schalk, A. Schlogl, C. Vidaurre,
        S. Waldert, and B. Blankertz, \Review of the bci competition iv," Frontiers in Neuroscience, vol. 6, p. 55,
        2012. Available: http://journal.frontiersin.org/article/10.3389/fnins.2012.00055

    """

    def __init__(self):
        super().__init__(title='Comp_IV_2a',
                         ch_names=['Fz', 'FC3', 'FC1', 'FCz',
                                   'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                                   'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                                   'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'],
                         fs=250,
                         doi='https://doi.org/10.3389/fnins.2012.00055',
                         url="http://bnci-horizon-2020.eu/database/data-sets/001-2014"
                         )
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, mode='',
                 epoch_duration=2,
                 band=[4.0, 40.0],
                 order=3):
        '''
        '''        
        epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
        data_mode = {'train':'T', 'test':'E'}        
        data_files = glob.glob(path + f"/*{data_mode[mode]}.mat")              
        data_files.sort()      
        subjects = range(len(data_files))
        X = []
        Y = []

        for subj in subjects:
            x, y = self._get_epoched(data_files[subj],
                                     epoch_duration,
                                     band,
                                     order)
            X.append(x)
            Y.append(y)
        
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    
    def load_raw_legacy(self, path=None, mode='',
                 epoch_duration=2,
                 band=[4.0, 40.0],
                 order=3):
        '''
        '''
        set_log_level(verbose=False)
        epoch_duration = np.round(
            np.array(epoch_duration) * self.fs).astype(int)
        labels_folder = path + '/true_labels'

        if mode == 'train':
            data_files = glob.glob(path + '/*T.gdf')
            labels_files = glob.glob(labels_folder + '/*T.mat')
        elif mode == 'test':
            data_files = glob.glob(path + '/*E.gdf')
            labels_files = glob.glob(labels_folder + '/*E.mat')

        data_files.sort()
        labels_files.sort()

        subjects = range(len(data_files))
        X = []
        Y = []

        for subj in subjects:
            x, y = self._get_epoched(data_files[subj],
                                     labels_files[subj],
                                     epoch_duration,
                                     band,
                                     order)
            X.append(x)
            Y.append(y)

        samples, channels, trials = X[0].shape
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def generate_set(self, load_path=None, epoch=2,
                     band=[4., 40.],
                     order=3,
                     download=False,
                     save=True,
                     fname=None,
                     save_folder=None):
        '''
        '''
        if download:
            self.download_raw(load_path)
            
        self.epochs, self.y = self.load_raw(load_path, 'train',
                                            epoch, band,
                                            order
                                            )

        self.test_epochs, self.test_y = self.load_raw(load_path,
                                                      'test', epoch,
                                                      band, order
                                                      )

        self.paradigm = self._get_paradigm()
        if save:
            self.save_set(save_folder, fname)        

    def get_path(self):
        NotImplementedError

    def download_raw(self, store_path=None):
        mode = ['train', 'test']
        for m in mode:
            links = self._files_link(m)
            [download_file(l, store_path) for l in links]        

    def _get_epoched(self, data_file, dur, band, order):
        '''
        '''
        raw = loadmat(data_file)
        raw = raw['data']
        epochs, ys = [], []

        subj_gender 	= raw[0,0][0,0][6].squeeze()
        subj_age 		= raw[0,0][0,0][7].squeeze()
        subj_id = data_file.split('/')[-1].split('.mat')[0].split('A')[1][:2]
        mode = data_file.split('/')[-1].split('.mat')[0].split('A')[1][-1]
        valid_sessions = 3         
        if subj_id == '04' and mode == 'T':
            valid_sessions = 1
        for ii in range(valid_sessions, raw.shape[1]):
            raw_signal 		= raw[0,ii][0,0][0][:,:22]
            raw_pos 	    = raw[0,ii][0,0][1].squeeze()
            y 		        = raw[0,ii][0,0][2].squeeze()
            # classes 		= raw[0,ii][0,0][4].squeeze()
            signal = bandpass(raw_signal, band, self.fs, order)
            ys.append(y)
            epochs.append(eeg_epoch(signal, dur, raw_pos))        

        raw_subject_info = {
            'id': subj_id,
            'gender': subj_gender,
            'age': subj_age,
            'handedness': '',
            'medication': '',
        }

        self.subjects.append(self._get_subjects(raw_subject_info))
        epochs = np.concatenate(epochs, axis=-1)
        ys = np.concatenate(ys)        
        return epochs, ys
    
    def _get_epoched_legacy(self, data_file, label_file,
                     dur, band, order):
        '''
        '''
        Left = 769
        Right = 770
        Foot = 771
        Tongue = 772
        Unkown = 783
        raw = read_raw_edf(data_file)
        lb = loadmat(label_file)
        y = lb['classlabel'].ravel()

        # keep EEG channels only, multiple by gain
        signal = np.divide(
            raw.get_data()[:22, :].T, raw._raw_extras[0]['units'][:22]).T
        # get events
        events_raw = raw._raw_extras[0]['events']
        events_pos = events_raw[1]
        events_desc = events_raw[2]
        ev_idx = (events_desc == Left) | (events_desc == Right) | (events_desc == Foot) | (events_desc == Tongue) | (events_desc == Unkown)
        ev_desc = events_desc[ev_idx]
        ev_pos = events_pos[ev_idx]
        # filter
        signal = bandpass(signal, band, self.fs, order=order)
        # epoch
        epochs = eeg_epoch(signal.T, dur, ev_pos)
        self.subjects.append(self._get_subjects(
            raw._raw_extras[0]['subject_info']))
        return epochs, y

    def _files_link(self, mode='train'):
        data_mode = {'train':'T', 'test':'E'}
        return [f"{self.url}/A0{n}{data_mode[mode]}.mat" for n in range(1,10)]      

    def _get_subjects(self, raw_subject_info):
        '''
        '''
        return Subject(id=raw_subject_info['id'],
                       gender=raw_subject_info['gender'],
                       age=raw_subject_info['age'],
                       handedness=raw_subject_info['handedness'],
                       condition=raw_subject_info['medication']
                       )

    def _get_paradigm(self):
        '''
        '''
        return MotorImagery(title='Comp_IV_2a',
                            stimulation=3000,
                            break_duration=2000,
                            repetition=72, stimuli=4,
                            phrase=''
                            )
