from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass, eeg_epoch
import numpy as np
import glob
import h5py


class MultiDay(DataSet):
    """
    Ga-Young Choi, Chang-Hee Han, Young-Jin Jung, Han-Jeong Hwang,
    A multi-day and multi-band dataset for a steady-state visual-evoked
    potential–based brain-computer interface, GigaScience, Volume 8,
    Issue 11, November 2019, giz133, https://doi.org/10.1093/gigascience/giz133
    """

    def __init__(self):
        super().__init__(title='Multiday',
                         ch_names=['Fp1', 'Fp2', 'AF4', 'AF3', 'F5',
                                   'Fz', 'FC1', 'FC5', 'F6', 'FC2',
                                   'FC6', 'C4', 'Cz', 'C3', 'CP1',
                                   'CP2', 'CP6', 'P8', 'P4', 'Pz',
                                   'POz', 'PO4', 'PO8', 'O2', 'Oz',
                                   'O1', 'PO3', 'P3', 'CP5', 'P7',
                                   'PO7', 'T7', 'T8'],
                         fs=200,
                         doi='https://doi.org/10.1093/gigascience/giz133')
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def generate_set(self, load_path=None, ch=None, epoch=[0, 6],
                     band=[[38., 44.], [3., 9.], [18., 24.]], order=6,
                     save_folder=None, fname=None,
                     augment=False, method='divide',
                     slide=0.1):
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
        epoch : list
            epoch window start and end in seconds relative to trials' onset
            default : [0, 6]
        band : list of lists
            band-pass filter frequencies for each frequency range, low-freq and high-freq
            default : [38., 44.] High
                      [3.0, 9.0] Low
                      [18., 24.] Mid
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
        franges = ['HIGH', 'LOW', 'MID']
        for i, rng in enumerate(franges):
            epochs, y = self.load_raw(load_path, ch,
                                      'train', rng,
                                      epoch, band[i],
                                      order, augment,
                                      method, slide
                                      )

            test_epochs, test_y = self.load_raw(load_path, ch,
                                                'test', rng,
                                                epoch, band[i],
                                                order, augment,
                                                method, slide
                                                )
            # self.subjects = self._get_subjects(n_subjects=29)
            subjects = self._get_subjects(n_subjects=30)
            self.paradigm = self._get_paradigm(frange=rng)
            events = self._get_events(y)
            test_events = self._get_events(test_y)
            if ch:
                # _ = self.select_channels(ch)
                indexes = self._get_channels(ch)
                ch_names = [self.ch_names[cc] for cc in indexes]
            else:
                ch_names = self.ch_names
            if fname:
                fname = f'{fname}_{rng}_'
            else:
                fname = f'{self.title}_{rng}_'
            #
            dataset = MultiDay()
            dataset.title = f'{self.title}_{rng}'
            dataset.ch_names = ch_names
            dataset.epochs = epochs
            dataset.y = y
            dataset.test_epochs = test_epochs
            dataset.test_y = test_y
            dataset.subjects = subjects
            dataset.paradigm = self.paradigm
            dataset.events = events
            dataset.test_events = test_events
            dataset.save_set(save_folder, fname=fname)
            #

    def load_raw(self, path=None, ch=None, mode='', frange='',
                 epoch_duration=[0, 6], band=[3., 9.], order=6,
                 augment=False, method='divide', slide=0.1):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str
            raw data folder path
        ch : list, optional
            default : None, keep all channels
        mode : str
            data acquisition session mode: 'train' or 'test'
        frange: str
            frequency range to select from the three available:
                - Low : 5.0, 5.5, 6.0, 6.5
                - Mid : 21.0, 21.5, 22.0, 22.5
                - High : 40.0, 40.5, 41.0, 41.5
        epoch_duration : list
            epoch window start and end in seconds relative to trials' onset
            default : [0, 6]
        band : list
            band-pass filter frequencies, low-freq and high-freq
            default : [3., 9.]
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
        y : nd array (subjects x trials)
            class labels for the entire set or train/test phase
        """
        if ch:
            # channels = self.select_channels(ch)
            channels = self._get_channels(ch)
        else:
            channels = range(33)

        if isinstance(epoch_duration, list):
            epoch_duration = (np.array(epoch_duration) * self.fs).astype(int)
        else:
            epoch_duration = (np.array([0, epoch_duration]) * self.fs).astype(int)

        if mode == 'train':
            session = '/Day1'
        else:
            session = '/Day2'

        subjects_list = sorted(glob.glob(path + '/S*'))
        X, Y = [], []
        # records = 6
        records = 2
        stimulation = 6 * self.fs
        for subj in subjects_list:
            k = 0
            x_subj, y_subj = [], []
            # cnt_list = sorted(glob.glob(subj+session+'/cnt*'))
            # mrk_list = sorted(glob.glob(subj+session+'/mrk*'))
            cnt_list = sorted(glob.glob(f'{subj}{session}/cnt_{frange}*'))
            mrk_list = sorted(glob.glob(f'{subj}{session}/mrk_{frange}*'))
            for i in range(records):
                cnt = h5py.File(cnt_list[i], 'r')
                mrk = h5py.File(mrk_list[i], 'r')
                # raw continuous EEG (samples x channels)
                x = cnt['cnt/x'][()].T
                y_orig = mrk['mrk/event/desc'][()].astype(int).squeeze()
                markers_orig = np.around(mrk['mrk/time'][()]).astype(int).squeeze()
                # x = x[:, :33]  # keeps only EEG channels
                x = x[:, channels]
                x = bandpass(x, band, self.fs, order)
                only_stimulations = y_orig != 5
                '''
                if 2 <= i < 4:
                    k = 4
                elif i >= 4:
                    k = 8
                '''
                # y = y_orig[only_stimulations] + k
                y = y_orig[only_stimulations]
                markers = markers_orig[only_stimulations]
                if augment:
                    v = self._get_augmented_cnt(x, epoch_duration, markers, stimulation, slide, method)
                    eeg = np.concatenate(v, axis=2)
                    y = np.tile(y, len(v))
                    del v
                else:
                    eeg = eeg_epoch(x, epoch_duration, markers)
                #
                del x
                cnt.close()
                mrk.close()
                x_subj.append(eeg)
                y_subj.append(y)

            X.append(np.concatenate(x_subj, axis=-1))
            Y.append(np.concatenate(y_subj, axis=-1))

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y

    def get_path(self):
        NotImplementedError

    def _get_events(self, y):
        """Attaches the experiments paradigm frequencies to
        class labels

        Parameters
        ----------
        y : nd array (subjects x trials)
            class labels

        Returns
        -------
        ev : nd array (subjects x trials)
        """
        ev = []
        for i, _ in enumerate(y):
            events = np.empty(y[i].shape, dtype=object)
            for l, _ in enumerate(self.paradigm.frequencies):
                ind = np.where(y[i] == l + 1)
                events[ind[0]] = self.paradigm.frequencies[l]
            ev.append(events)
        return ev

    def _get_paradigm(self, frange=''):
        """Get experimental paradigm

        Parameters
        ----------
        frange : str
            frequency range to select from the three available:
                - Low : 5.0, 5.5, 6.0, 6.5
                - Mid : 21.0, 21.5, 22.0, 22.5
                - High : 40.0, 40.5, 41.0, 41.5


        Returns
        -------
        Paradigm instance
        """
        ranges = {'high': 0, 'low': 1, 'mid': 2}
        frequencies = ['40.0', '40.5', '41.0', '41.5',
                       '5.0', '5.5', '6.0', '6.5',
                       '21.0', '21.5', '22.0', '22.5',
                       ]
        freqs = np.arange(0, 4) + 4 * ranges[frange.lower()]
        fq = [frequencies[i] for i in freqs.tolist()]
        return SSVEP(title='SSVEP_ON_OFF', control='Sync',
                     stimulation=6000,
                     break_duration=6000, repetition=20,
                     stimuli=4, phrase='',
                     stim_type='LED',
                     frequencies=fq
                     )

    @staticmethod
    def h5ref_to_strings(hf, ref):
        """convert HDF5 reference object to String.

        Parameters
        ----------
        hf : HDF5 file
            an instance of an opened HDF5 file containing
            the group/dataset
        ref : HDF5 Dataset
            HDF5 dataset of strings

        Returns
        -------
        List of strings
        """
        shape = ref.value.shape
        if shape[1] > shape[0]:
            array = ref.value.T
        else:
            array = ref.value
        return [''.join(chr(i) for i in hf[obj[0]]) for obj in array]
