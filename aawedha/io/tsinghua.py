from aawedha.io.base import DataSet
from aawedha.paradigms.ssvep import SSVEP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
from scipy.io import loadmat
import numpy as np
import glob
import pickle
import os
import gzip
import re


class Tsinghua(DataSet):
    '''
    Tsinghua SSVEP sampled sinusoidal joint frequency-phase modulation (JFPM)
    [1} X. Chen, Y. Wang, M. Nakanishi, X. Gao, T. -P. Jung, S. Gao,
       "High-speed spelling with a noninvasive brain-computer interface",
       Proc. Int. Natl. Acad. Sci. U. S. A, 112(44): E6058-6067, 2015.
    '''

    def __init__(self):
        super().__init__(title='Tsinghua',
                        ch_names=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
                        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
                        'FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7',
                        'C5','C3','C1','Cz','C2','C4','C6','T8','M1',
                        'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
                        'TP8','M2','P7','P5','P3','P1','PZ','P2',
                        'P4','P6','P8','PO7','PO5','PO3','POz','PO4',
                        'PO6','PO8','CB1','O1','Oz','O2','CB2'],
                        fs=250,
                        doi='http://dx.doi.org/10.1073/pnas.1508080112'
                        )

    def load_raw(self, path=None, epoch_duration=1,
                 band=[5.0, 45.0], order=6, augment=False):
        '''
        '''
        list_of_files =  np.array(glob.glob(path+'/S*.mat'))
        indices =  np.array([int(re.findall(r'\d+', n)[0]) for n in list_of_files]) - 1
        onset = int(0.5*self.fs)
        epoch_duration =  np.round(np.array(epoch_duration+0.5) * self.fs).astype(int)           
            
        n_subjects = 35               
        
        X = []
        Y = []
        augmented = 0
        for subj in range(n_subjects):
            data = loadmat(list_of_files[indices==subj][0])
            eeg = data['data'].transpose((1,0,2,3))
            eeg = bandpass(eeg, band=band, fs=self.fs, order=order)
            if augment:
                # TODO
                augmented = 4
                v = [eeg[onset+(stride*self.fs):onset+(stride*self.fs)+epoch_duration, :, :, :] for stride in range(augmented)]
                eeg = np.concatenate(v, axis=2)
                samples, channels, targets, blocks = eeg.shape
                y = np.tile(np.arange(1, targets+1), (int(blocks/augmented),1))
                y = np.tile(y, (1,augmented))
                y = y.reshape((1,blocks*targets),order='F')
            else:
                eeg = eeg[onset:epoch_duration, :, :, :]                
                samples, channels, targets, blocks = eeg.shape
                y = np.tile(np.arange(1, targets+1), (1,blocks))
            
            X.append(eeg.reshape((samples, channels, blocks*targets),order='F'))
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y).squeeze()
        return X, Y


    def generate_set(self, load_path=None, epoch=1, band=[5.0,45.0],
                    order=6, save_folder=None, augment=False):
        '''
        '''
        self.epochs , self.y = self.load_raw(load_path, 
                                             epoch, band, order,
                                             augment)
        self.subjects = self._get_subjects(path=load_path)
        self.paradigm = self._get_paradigm()                                     
         # save dataset
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        fileName = save_folder + '/tsinghua.pkl'
        f = gzip.open(fileName, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()

    
    def load_set(self, fileName=None):
        '''
        '''
        if os.path.exists(fileName):
            f = gzip.open(fileName, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data


    def _get_subjects(self, n_subject=0, path=None):
        '''
        '''
        sub_file = path + '/Sub_info.txt'
        f = open(sub_file, 'r')
        info = f.read().split('\n')[2:]
        f.close()
        return [Subject(id=s.split()[0],gender=s.split()[1],
                age=s.split()[2], handedness=s.split()[3]) 
                for s in info if len(s)>0]

    def _get_paradigm(self):
        '''
        '''
        return SSVEP(title='SSVEP_JFPM', stimulation=5000, break_duration=500, repetition=6,
                    stimuli=40, phrase='',
                    stim_type='Sinusoidal', frequencies=[8. ,  9. , 10. , 11. , 12. , 13. , 14. , 15. ,  8.2,  9.2, 
                                                         10.2, 11.2, 12.2, 13.2, 14.2, 15.2,  8.4,  9.4, 10.4, 11.4, 
                                                         12.4, 13.4, 14.4, 15.4,  8.6,  9.6, 10.6, 11.6, 12.6, 13.6, 
                                                         14.6, 15.6,  8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8], 
                    phase=[0., 1.57079633, 3.14159265, 4.71238898, 0. ,
                           1.57079633, 3.14159265, 4.71238898, 1.57079633, 3.14159265,
                           4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898,
                           0.        , 3.14159265, 4.71238898, 0.        , 1.57079633,
                           3.14159265, 4.71238898, 0.        , 1.57079633, 4.71238898,
                           0.        , 1.57079633, 3.14159265, 4.71238898, 0.        ,
                           1.57079633, 3.14159265, 0.        , 1.57079633, 3.14159265,
                           4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898] 
                    )


    def get_path(self):
        NotImplementedError

                