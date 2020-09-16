from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import glob


class Perturbations(DataSet):
    """İşcan Z, Nikulin VV (2018) Steady state visual evoked potential (SSVEP)
    based brain-computer interface (BCI) performance under different
    perturbations. PLoS ONE 13(1): e0191673.
    """

    def __init__(self):
        super().__init__(title='Perturbations',
                         ch_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1',
                                   'C3', 'T7', 'EOGL', 'CP5', 'CP1', 'Pz',
                                   'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                                   'EOGC', 'CP6', 'CP2', 'Cz', 'C4', 'T8',
                                   'EOGR', 'FC6', 'FC2', 'F4', 'F8', 'FP2',
                                   'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7',
                                   'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3',
                                   'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                                   'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8',
                                   'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2',
                                   'AF4', 'AF8', 'BIP1'],
                         fs=1000,
                         doi='https://doi.org/10.1371/journal.pone.0191673')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, ch=None, downsample=4, mode='', epoch_duration=3,
                 band=[5.0, 45.0], order=6, augment=False,
                 method='divide', slide=0.1):

        ep = epoch_duration
        X, Y = [], []
        # augmented = 0
        factor = 1
        if downsample:
            factor = 4

        if ch:
            eeg_channels = self.select_channels(ch)
        else:
            eeg_channels = self._get_eeg_channels()
        # onset = 0
        if mode == 'train':
            path = path + '/*mat'
            mode_key = 'circle_order'
        else:
            path = path + '/*control.mat'
            mode_key = 'subject_selection_vector'
        files_list = glob.glob(path)
        stimulation = 3
        for f in files_list:
            data = loadmat(f)
            x = data['data_matrix'][eeg_channels, ::factor, :].transpose((1, 0, 2))
            y = data[mode_key].squeeze()
            '''
            if mode == 'train':
                y = data['circle_order'].squeeze()
            else:
                y = data['subject_selection_vector'].squeeze()
            '''
            x = bandpass(x, band=band, fs=self.fs, order=order)
            if augment:
                # stimulation = 3
                # augmented = np.floor(stimulation * self.fs / epoch_duration).astype(int)
                # strides = list(np.arange(0, stimulation, ep))
                # v = [x[onset + int(s * self.fs):onset + int(s * self.fs) + epoch_duration, :, :] for s in strides]
                # v = self._get_augmented(x, ep, method, slide)
                v = self._get_augmented_epoched(x, ep, stimulation, slide=slide, method=method)
                x = np.concatenate(v, axis=2)
                # y = np.tile(y, augmented)
                y = np.tile(y, len(v))
                del v
            else:
                epoch_duration = np.round(np.array(epoch_duration) * self.fs).astype(int)
                x = x[:epoch_duration, :, :]

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y

    def generate_set(self, load_path=None, ch=None,
                     downsample=4,
                     epoch=1,
                     band=[5.0, 45.0],
                     order=6, save_folder=None,
                     augment=False,
                     method='divide',
                     slide=0.1):
        '''
        '''
        offline = load_path + '/OFFLINE'
        online = load_path + '/ONLINE'
        if downsample:
            self.fs = self.fs // int(downsample)

        self.epochs, self.y = self.load_raw(offline, ch, downsample,
                                            'train',
                                            epoch, band,
                                            order, augment,
                                            method, slide
                                            )

        self.test_epochs, self.test_y = self.load_raw(online, ch, downsample,
                                                      'test', epoch,
                                                      band, order, augment,
                                                      method, slide
                                                      )
        self.subjects = self._get_subjects(n_subjects=24)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events(self.y)
        self.test_events = self._get_events(self.test_y)
        if not ch:
            eeg_channels = self._get_eeg_channels()
            self.ch_names = [self.ch_names[i] for i in eeg_channels]
        self.save_set(save_folder)

    def _get_augmented(self, eeg, epoch, method='divide', slide=0.1):
        """Segment a single epoch of a continuous signal into augmented epochs

         Parameters
        ----------
        eeg : ndarray
            epoched EEG signal : samples x channels x blocks x targets
        epoch : array
            onset of epoch length
        slide : float
            stride for sliding window segmentation method in seconds
        method : str
            segmentation method :
                - divide : divide epochs by length of epochs, no overlapping
                - slide : sliding window by a stride, overlapping
        Returns
        -------
        v : list
            list of segmented epochs belonging to the same class/target
        """
        onset = 0
        epoch_duration = np.round(np.array(epoch) * self.fs).astype(int)
        stimulation = 3
        v = []

        if method == 'divide':
            strides = range(np.floor(stimulation * self.fs / epoch_duration).astype(int))
            v = [eeg[onset + int(s * self.fs):onset + int(s * self.fs) + epoch_duration, :, :] for s in strides]
        elif method == 'slide':
            augmented = int((stimulation - epoch) // slide) + 1
            ops = range(augmented)
            slide = int(slide * self.fs)
            v = [eeg[onset + slide * s:onset + slide * s + epoch_duration, :, :] for s in ops]

        return v

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

    @staticmethod
    def _get_paradigm():
        return SSVEP(title='SSVEP_ON_OFF', control='Sync',
                     stimulation=3000,
                     break_duration=1000, repetition=25,
                     stimuli=4, phrase='',
                     stim_type='ON_OFF',
                     frequencies=['15.', '12.', '8.57', '5.45'],
                     )

    def _get_eeg_channels(self):
        exclude = ['EOGL', 'EOGC', 'EOGR', 'BIP1']
        eeg_channels = [i for i in range(64) if self.ch_names[i] not in exclude]
        return eeg_channels
