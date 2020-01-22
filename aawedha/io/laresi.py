from aawedha.io.base import DataSet
from aawedha.paradigms.hybrid import HybridLARESI
from aawedha.paradigms.subject import Subject
from aawedha.paradigms.ssvep import SSVEP
from aawedha.analysis.preprocess import bandpass
from aawedha.analysis.preprocess import eeg_epoch
from abc import abstractmethod
import numpy as np
import pandas as pd
import glob
import pickle
import os

OV_ExperimentStart = 32769
OV_ExperimentStop = 32770
Base_Stimulations = 33024
OV_Target = 33285
OV_NonTarget = 33286
OV_TrialStart = 32773
OV_TrialStop = 32774


class LaresiHybrid:
    """
        LARESI Hybrid ERP-SSVEP Brain computer interface
        Reference:
        [] coming soon...
    """

    def __init__(self):
        self.title = 'Laresi_hybrid'
        self.ch_names = ['Pz', 'PO5', 'PO3', 'POz', 'PO4',
                         'PO6', 'O1', 'Oz', 'O2', 'Fz',
                         'Cz', 'P3', 'P4', 'FCz']
        self.fs = 512
        self.doi = ''
        self.erp_set = LaresiERP(
            title='Hybrid_ERP', ch_names=self.ch_names, fs=self.fs)
        self.ssvep_set = LaresiSSVEP(
            title='Hybrid_SSVEP', ch_names=self.ch_names, fs=self.fs)
        self.paradigm = HybridLARESI()

    def load_raw(self, path=None):
        '''
        '''
        files_list = sorted(glob.glob(path + '/s*'))
        n_subjects = 1
        cnts, infos = [], []

        for subj in range(n_subjects):
            train_session = glob.glob(files_list[subj] + '/*calib*.csv')
            test_session = glob.glob(files_list[subj] + '/*online*.csv')
            cnt, info = self._read_raw(train_session[0])
            cnt_test, info_test = self._read_raw_online(test_session[0])
        cnts = [cnt, cnt_test]
        infos = [info, info_test]
        return cnts, infos  # cnt: EEG, info : dict

    def generate_set(self, load_path=None, save_folder=None,
                     epoch=[0., 0.7, 0., 1.0], band=[1, 10, 5, 45],
                     order=[2, 6]):
        '''
        '''
        n_subjects = 1  # 1 for now
        for subj in range(n_subjects):
            cnts, infos = self.load_raw(load_path)
            self.erp_set.generate_set(cnts, [infos[0]['ERP'], infos[1]],
                                      epoch=[epoch[0], epoch[1]],
                                      band=[band[0], band[1]], order=order[0])
            self.ssvep_set.generate_set(cnts, [infos[0]['SSVEP'], infos[1]],
                                        epoch=[epoch[2], epoch[3]],
                                        band=[band[2], band[3]], order=order[1])

        self.erp_set._cat_lists()
        self.ssvep_set._cat_lists()
        self.ssvep_set.events = self.ssvep_set._get_events(self.ssvep_set.y)
        self.ssvep_set.test_events = self.ssvep_set._get_events(
            self.ssvep_set.test_y)
        self.save_set(save_folder)

    def save_set(self, save_folder=None):
        '''
        '''
        # save dataset
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        if not self.title:
            self.title = 'unnamed_set'

        fname = save_folder + '/' + self.title + '.pkl'
        print(f'Saving dataset {self.title} to destination: {fname}')
        f = open(fname, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_set(self, file_name=None):
        '''
        '''
        if os.path.exists(file_name):
            f = open(file_name, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data

    def _read_raw(self, path=None):
        '''
        '''
        cnt = []
        info = {}
        if path:
            cnt, raw_desc, raw_pos = self._convert_raw(path)
            start_interv = raw_pos[raw_desc == OV_ExperimentStart]
            end_interv = raw_pos[raw_desc == OV_ExperimentStop]
            if end_interv.size > 1:
                # Hybrid paradigm
                start2 = start_interv[start_interv > end_interv[0]][0]
                starts = np.array([start_interv[0], start2])
            else:
                starts = start_interv[0]
            info = self._get_info(raw_desc, raw_pos, starts, end_interv)

        return cnt, info

    def _read_raw_online(self, path=None):
        '''
        '''
        cnt_test = []
        info_test = {}
        if path:
            cnt_test, raw_desc, raw_pos = self._convert_raw(path)
            info_test['raw_desc'] = raw_desc
            info_test['raw_pos'] = raw_pos

        return cnt_test, info_test

    def _convert_raw(self, path=None):
        '''
        '''
        markers = {}
        raw = pd.read_csv(path)
        cnt = raw[self.ch_names].to_numpy()  # samples x channels
        raw_desc = self._get_events(raw, 'Event Id')
        raw_pos = self._get_events(raw, 'Event Date')
        return cnt, raw_desc, raw_pos

    def _get_info(self, raw_desc, raw_pos, starts, ends):
        '''
        '''

        info = {}
        for i in range(len(starts)):

            idx = np.logical_and(raw_pos >= starts[i], raw_pos <= ends[i])
            desc = raw_desc[idx]
            pos = raw_pos[idx]
            evs = np.unique(desc)
            stimuli = np.sum(np.logical_and(
                evs > Base_Stimulations, evs < OV_Target))
            id_stim = np.logical_and(
                desc > Base_Stimulations, desc <= Base_Stimulations + stimuli)
            desc = desc[id_stim] - Base_Stimulations
            pos = np.floor(pos[id_stim] * self.fs).astype(int)

            if np.any(evs == OV_Target):  # erp
                info['ERP'] = {}
                y = raw_desc[np.logical_or(
                    raw_desc == OV_Target, raw_desc == OV_NonTarget)]
                y[y == OV_Target] = 1
                y[y == OV_NonTarget] = 0
                info['ERP']['desc'] = desc
                info['ERP']['pos'] = pos
                info['ERP']['y'] = y
                info['ERP']['session_interval'] = (
                    np.floor([starts[i], ends[i]]) * self.fs).astype(int)
            else:  # ssvep
                y = desc
                info['SSVEP'] = {}
                info['SSVEP']['desc'] = desc
                info['SSVEP']['pos'] = pos
                info['SSVEP']['y'] = y
                info['SSVEP']['session_interval'] = (
                    np.floor([starts[i], ends[i]]) * self.fs).astype(int)

            # session_interval = np.floor( [starts, ends] ) * self.fs # begin,
            # end of session
        return info

    def _get_events(self, dataframe, key):
        events_id = dataframe[key].notna()
        events = dataframe[key].loc[events_id]
        events = events.to_numpy()
        ev = [elm.split(':') for elm in events]
        ev = np.array(list(pd.core.common.flatten(ev)), dtype=float)
        return ev

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S' + str(s), gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]


# single paradigm class
class LaresiEEG(DataSet):

    def __init__(self, title='', ch_names=[], fs=None, doi=''):
        super().__init__(title=title, ch_names=ch_names, fs=fs, doi=doi)
        self.test_epochs = []
        self.test_y = []
        self.test_events = []

    def load_raw(self, cnts=[], infos=[], epoch_duration=[0, .7],
                 band=[1., 10.], order=2,
                 augment=False):
        '''
        '''
        epochs = self._get_epochs(
            cnts[0], infos[0], epoch_duration, band, order)
        test_epochs, test_y = self._get_epochs_test(
            cnts[1], infos[1], epoch_duration, band, order)
        return epochs, test_epochs, test_y

    def generate_set(self,
                     cnts=[],
                     infos=[],
                     epoch=[0, .7],
                     band=[1., 10],
                     order=2,
                     augment=False):
        '''
        '''
        epochs, test_epochs, y_test = self.load_raw(cnts, infos,
                                                    epoch, band,
                                                    order, augment
                                                    )
        self.epochs.append(epochs)
        self.y.append(infos[0]['y'])
        self.test_epochs.append(test_epochs)
        self.test_y.append(y_test)
        self.subjects = []
        self.paradigm = self._get_paradigm()

    def get_path(self):
        return NotImplementedError

    def _get_epochs(self, cnt, info, epoch, band, order):
        '''
        '''
        epoch = np.round(np.array(epoch) * self.fs).astype(int)
        signal = cnt[info['session_interval']
                     [0]:info['session_interval'][1], :]
        signal = bandpass(signal, band, self.fs, order)
        cnt[info['session_interval'][0]:info['session_interval'][1], :] = signal
        eps = eeg_epoch(cnt, epoch, info['pos'])
        # self.events = info['desc']
        return eps

    @abstractmethod
    def _get_epochs_test(self):
        pass

    def _cat_lists(self):
        '''
        '''
        self.epochs = np.array(self.epochs)
        self.y = np.array(self.y)
        self.test_epochs = np.array(self.test_epochs)
        self.test_y = np.array(self.test_y)

    def _get_paradigm(self):
        '''
        '''
        return []


class LaresiERP(LaresiEEG):

    def _get_epochs_test(self, cnt=None, markers={},
                         epoch=[0., 0.7], band=[1, 10], order=2):
        '''
        '''
        raw_desc = markers['raw_desc']
        raw_pos = markers['raw_pos']

        start = raw_pos[raw_desc == OV_TrialStart]
        end = raw_pos[raw_desc == OV_TrialStop]
        erp_start = np.round(start[::2] * self.fs).astype(int)
        erp_end = np.round(end[::2] * self.fs).astype(int)

        evs = np.unique(raw_desc)
        stimuli = np.sum(np.logical_and(
            evs > Base_Stimulations, evs < OV_Target))
        id_stim = np.logical_and(
            raw_desc > Base_Stimulations, raw_desc <= Base_Stimulations + stimuli)
        stim_mrk = np.round(raw_pos[id_stim] * self.fs).astype(int)
        stim_m = raw_desc[id_stim]

        erp_tr = np.logical_or(raw_desc == OV_Target, raw_desc == OV_NonTarget)
        erp_mrk = np.round(raw_pos[erp_tr] * self.fs).astype(int)

        y_erp = raw_desc[erp_tr]
        y_erp[y_erp == OV_Target] = 1
        y_erp[y_erp == OV_NonTarget] = 0

        erp_epochs = []
        desc = []
        epoch = np.round(np.array(epoch) * self.fs).astype(int)

        for tr in range(len(erp_start)):
            # filter, epoch, append
            desc_idx = np.logical_and(
                stim_mrk >= erp_start[tr], stim_mrk <= erp_end[tr])
            desc.append(stim_m[desc_idx] - Base_Stimulations)

            idx = np.logical_and(
                erp_mrk >= erp_start[tr], erp_mrk <= erp_end[tr])
            mrk = erp_mrk[idx] - erp_start[tr]
            erp_signal = bandpass(
                cnt[erp_start[tr]:erp_end[tr], :], band, self.fs, order)
            erp_epochs.append(eeg_epoch(erp_signal, epoch, mrk))

        samples, channels, trials = erp_epochs[0].shape
        blocks = len(erp_epochs)
        eps = np.array(erp_epochs).transpose((1, 2, 3, 0)).reshape(
            (samples, channels, trials * blocks))

        self.test_events = desc

        return eps, y_erp


class LaresiSSVEP(LaresiEEG):

    def _get_epochs_test(self, cnt=None, markers={},
                         epoch=[0., 1.0], band=[5, 45], order=6):
        '''
        '''
        raw_desc = markers['raw_desc']
        raw_pos = markers['raw_pos']

        start = raw_pos[raw_desc == OV_TrialStart]
        end = raw_pos[raw_desc == OV_TrialStop]
        ssvep_start = start[1::2]
        ssvep_end = end[1::2]
        ev_desc = []
        ev_pos = []
        base = Base_Stimulations
        for i in range(len(ssvep_start)):
            idx1 = np.logical_and(
                raw_pos > ssvep_start[i], raw_pos < ssvep_end[i])
            # 10 is an arbitary number
            idx2 = np.logical_and(raw_desc > base, raw_desc <= base + 10)
            idx = np.logical_and(idx1, idx2)
            ev_desc.append(raw_desc[idx][0] - base)
            ev_pos.append(np.round(raw_pos[idx][0] * self.fs).astype(int))

        ev_desc = np.array(ev_desc)  # y
        ev_pos = np.array(ev_pos)
        ssvep_epochs = []

        epoch = np.round(np.array(epoch) * self.fs).astype(int)

        ssvep_start = np.round(ssvep_start * self.fs).astype(int)
        ssvep_end = np.round(ssvep_end * self.fs).astype(int)

        for tr in range(len(ev_desc)):
            # filter, epoch, append
            mrk = ev_pos[tr] - ssvep_start[tr]
            ssvep_signal = bandpass(
                cnt[ssvep_start[tr]:ssvep_end[tr], :], band, self.fs, order)
            ssvep_epochs.append(
                eeg_epoch(ssvep_signal, epoch, np.array([mrk])).squeeze())

        ssvep_epochs = np.array(ssvep_epochs).transpose((1, 2, 0))
        return ssvep_epochs, ev_desc

    def _get_events(self, y):
        '''
        '''
        events = np.empty(y.shape, dtype=object)
        for i in range(events.shape[0]):
            for l in range(len(self.paradigm.frequencies)):
                ind = np.where(y[i, :] == l+1)
                events[i, ind[0]] = self.paradigm.frequencies[l]
        return events

    def _get_paradigm(self):
        return SSVEP(title='SSVEP_LARESI', stimulation=1000,
                     break_duration=2000, repetition=15,
                     stimuli=5, phrase='', stim_type='Sinusoidal',
                     frequencies=['idle', '14', '12', '10', '8'],
                     )
