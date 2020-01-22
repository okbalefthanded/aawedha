from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass, eeg_epoch
import numpy as np
import pickle
import glob
import gzip


class Exoskeleton(DataSet):
    """
    SSVEP-based BCI of subjects operating an upper limb exoskeleton during a shared control

    References :
    [1] Emmanuel K. Kalunga, Sylvain Chevallier, Olivier Rabreau, Eric Monacelli. Hybrid interface:
    Integrating BCI in Multimodal Human-Machine Interfaces. IEEE/ASME International Conference
    on Advanced Intelligent Mechatronics (AIM), 2014, Besancon, France.
    [2] Emmanuel Kalunga, Sylvain Chevallier, Quentin Barthelemy. Data augmentation in
    Riemannian space for Brain-Computer Interfaces, STAMLINS (ICML workshop),
    2015, Lille, France.
    [3] Emmanuel K. Kalunga, Sylvain Chevallier, Quentin Barthelemy. Online SSVEP-based BCI
    using Riemannian Geometry. Neurocomputing, 2016. arXiv research report on arXiv:1501.03227.

    """

    def __init__(self):
        super().__init__(title='Exoskeleton_SSVEP',
                         ch_names=['Oz', 'O1', 'O2', 'PO3',
                                   'POz', 'PO7', 'PO8', 'PO4'],
                         fs=256,
                         doi='http://dx.doi.org/10.1109/AIM.2014.6878132')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, path=None, mode='', epoch_duration=[2, 5],
                 band=[5.0, 45.0], order=6, augment=False):
        '''
        '''
        files_list = sorted(glob.glob(path + '/s*'))
        n_subjects = 12
        X = []
        Y = []
        for subj in range(n_subjects):
            session = glob.glob(files_list[subj] + '/*.pz')
            if mode == 'train':
                records = np.arange(0, len(session) - 1).tolist()
            elif mode == 'test':
                records = [len(session) - 1]
            x, y = self._get_epoched(session, records,
                                     epoch_duration, band,
                                     order, augment)
            X.append(x)
            Y.append(y)

        # X = self.flatten(X)
        # Y = self.flatten(Y)
        # X = np.array(X)
        # Y = np.array(Y)
        return X, Y

    def generate_set(self, load_path=None, epoch=[2, 5],
                     band=[5., 45.],
                     order=6,
                     save_folder=None,
                     augment=False):
        '''
        '''
        self.epochs, self.y = self.load_raw(load_path, 'train',
                                            epoch, band,
                                            order, augment
                                            )

        self.test_epochs, self.test_y = self.load_raw(load_path,
                                                      'test', epoch,
                                                      band, order, augment
                                                      )

        self.subjects = self._get_subjects(n_subjects=12)
        self.paradigm = self._get_paradigm()
        self.events = self._get_events(self.y)
        self.test_events = self._get_events(self.test_y)
        self.save_set(save_folder)

    def _get_epoched(self, files=[], records=[],
                     epoch=[2, 5], band=[5., 45.],
                     order=6, augment=False):
        '''
        '''
        OVTK_StimulationId_VisualStimulationStart = 32779
        labels = [33024, 33025, 33026, 33027]
        if isinstance(epoch, list):
            epoch = np.array(epoch).astype(int) * self.fs
        else:
            epoch = np.array([0, epoch]).astype(int) * self.fs
        x = []
        y = []
        for sess in records:
            f = gzip.open(files[sess], 'rb')
            df = pickle.load(f, encoding='latin1')
            raw_signal = df['raw_signal'] * 1000  # samples, channels
            event_pos = df['event_pos'].reshape((df['event_pos'].shape[0]))
            event_type = df['event_type'].reshape((df['event_type'].shape[0]))
            desc_idx = np.logical_or.reduce(
                [event_type == lb for lb in labels])
            desc = event_type[desc_idx]
            # y.append(desc - 33023)
            pos = event_pos[event_type ==
                            OVTK_StimulationId_VisualStimulationStart]
            raw_signal = bandpass(raw_signal, band, self.fs, order)
            if augment:
                stimulation = 5 * self.fs
                augmented = np.floor(
                    stimulation / np.diff(epoch))[0].astype(int)
                v = [eeg_epoch(raw_signal, epoch + np.diff(epoch) * i, pos)
                     for i in range(augmented)]
                epchs = np.concatenate(v, axis=2)
                y.append(np.tile(desc - 33023, augmented))
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
        '''
        '''
        ev = []
        for i in range(len(y)):
            events = np.empty(y[i].shape, dtype=object)
            for l in range(len(self.paradigm.frequencies)):
                ind = np.where(y[i] == l+1)
                events[ind[0]] = self.paradigm.frequencies[l]
            ev.append(events)
        return ev

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S' + str(s), gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_LED', control='Async',
                     stimulation=5000,
                     break_duration=3000, repetition=8,
                     stimuli=3, phrase='',
                     stim_type='ON_OFF', frequencies=['idle', '13', '21', '17'],
                     )

    def flatten(self, list_of_lists=[]):
        return [item for sublist in list_of_lists for item in sublist]

    def get_path(self):
        NotImplementedError
