from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from aawedha.analysis.preprocess import bandpass
import numpy as np
import pandas as pd
import glob
import pickle
import os
import gzip

class Inria_ERN(DataSet):
    """
        Feedback Error-Related Negativity dataset [1]

        Reference :
        [1] P. Margaux, M. Emmanuel, D. Sébastien, B. Olivier, and M. Jérémie, Objective and subjective
        evaluation of online error correction during p300-based spelling," Advances in Human-
        Computer Interaction, vol. 2012, p. 4, 2012.
    """

    def __init__(self):
        super().__init__(title='Inria_ERN',
                        ch_names=['Fp1', 'Fp2', 'AF7','F3', 'AF4','F8','F7','F5','F3','F1',
                       'Fz', 'F2', 'F4', 'F6','F8','FT7','FC5','FC3','FC1',
                       'FCz', 'FC2', 'FC4', 'FC6', 'FT8','T7','C5','C3','C1',
                       'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz',
                       'CP2','CP4','CP6','TP8','P7', 'P5','P3','P1','Pz',
                       'P2','P4','P6','P8','PO7','POz','P08','O1','O2'], 
                       fs=200, 
                       doi = 'http://dx.doi.org/10.1155/2012/578295'
                       )      
      


    def load_raw(self, path=None, epoch_duration=1, band=[1.0, 40.0], order=5):
        """
        """
        list_of_files = glob.glob(path + 'Data_*.csv')
        list_of_files.sort() 

        n_subjects = 16
        epoch_duration = round(epoch_duration * self.fs) 
        channels = len(self.ch_names)
        epochs = 340
        X = []

        for f in list_of_files:
            sig = np.array(pd.io.parsers.read_csv(f))
            eeg = sig[:,1:-2]
            trigger = sig[:,-1]
            signal = bandpass(eeg, band, self.fs, order)
            idxFeedback = np.where(trigger==1)[0]

            for idx in idxFeedback:
                X.append(signal[idx:idx+epoch_duration, :])
            
            del sig # saving some RAM 

        X = np.array(X).transpose((2,1,0)).reshape((n_subjects, epoch_duration, channels, epochs))

        labels = np.genfromtxt(path + 'TrainLabels.csv',delimiter=',',skip_header=1)[:,1]
        Y = labels.reshape((n_subjects, epochs))
        return X, Y  # : 4-D array : (subject, samples, channels, epoch)

    def generate_set(self, load_path=None, epoch=1, band=[1.0, 40.0], 
                        order=5, save_folder=None):
        """
        """
        self.epochs, self.y = self.load_raw(load_path, epoch, band, order)
        self.subjects = self._get_subjects(n_subjects=16)
        self.paradigm = self._get_paradigm()
        
        # save dataset
        # save_folder = '/data/inria_ern'
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        fileName = save_folder + '/inria_ern.pkl'
        f = gzip.open(fileName, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)        
        f.close()

    def load_set(self, fileName=None):
        """
        """
        if os.path.exists(fileName):
            f = gzip.open(fileName, 'rb')
            data = pickle.load(f)
        else:
            raise FileNotFoundError
        f.close()
        return data  
        
    def get_path(self):
        NotImplementedError

    def _get_events(self):
        NotImplementedError

    def _get_subjects(self, n_subjects=0):
        return [Subject(id='S'+str(s),gender='M',age=0, handedness='')
                    for s in range(1, n_subjects+1)]

    def _get_paradigm(self):
        return ERP(title='ERP_ERN', stimulation=60, break_duration=50, repetition=12,
                    phrase='', flashing_mode='RC')

    

    
        

    


        
