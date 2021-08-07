from aawedha.io.base import DataSet
from aawedha.paradigms.erp import ERP
from aawedha.paradigms.subject import Subject
from aawedha.utils.network import download_file
from aawedha.utils.utils import unzip_files
from scipy.io import loadmat
import numpy as np
import glob



files = ['description.pdf',
         'online_study_1-7.zip',
         'online_study_8-13.zip',
         'sequence.mat'
         ]

class FreiburgOnline(DataSet):
    """
        Freiburg ERP online speller dataset [1]

        Reference:
        [1] Hübner D, Verhoeven T, Schmid K, Müller K-R, Tangermann M and Kindermans P-J.
        Learning from Label Proportions in Brain-Computer Interfaces: Online Unsupervised
        Learning with Guarantees. PLOS ONE. 2017
    """
    def __init__(self):
        super().__init__(title='Freiburg_ERP_online',
                         ch_names=['C3', 'C4', 'CP1', 'CP2',
                                   'CP5', 'CP6', 'Cz', 'F10',
                                   'F3', 'F4', 'F7', 'F8',
                                   'F9', 'FC1', 'FC2', 'FC5',
                                   'FC6', 'Fp1', 'Fp2', 'Fz',
                                   'O1', 'O2', 'P10', 'P3',
                                   'P4', 'P7', 'P8', 'P9',
                                   'Pz', 'T7', 'T8'],
                         fs=100,
                         doi='https://doi.org/10.1371/journal.pone.0175856',
                         url="https://zenodo.org/record/192684/files"
                         )

    def load_raw(self, path=None):
        """Read and process raw data into structured arrays

        Parameters
        ----------
        path : str,
            Dataset folder path

        Returns
        -------
        X : nd array (subjects x samples x channels x trials)
            EEG epochs
        Y : nd array (subjects x trials)
            epochs labels : 0/1 : 0 non target, 1 target
        """
        files_list = sorted(glob.glob(path + '/S*.mat'))
        n_subjects = 13
        X = []
        Y = []
        for subj in range(n_subjects):
            data = loadmat(files_list[subj])
            X.append(data['epo']['x'][0][0][20:, :, :])
            Y.append(data['epo']['y'][0][0].argmin(axis=0))
            del data

        samples, channels, trials = X[0].shape
        X = np.array(X).reshape(
            (n_subjects, samples, channels, trials), order='F')
        Y = np.array(Y).reshape((n_subjects, trials), order='F')

        return X, Y

    def generate_set(self, load_path=None, download=False,
                     save=True, save_folder=None,
                     fname=None):
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
        save : bool,
            it True save DataSet, default True.
        save_folder : str
            DataSet object saving folder path
        fname: str, optional
            saving path for file, specified when different versions of
            DataSet are saved in the same folder
            default: None
        """
        if download:
            self.download_raw(load_path)
        self.epochs, self.y = self.load_raw(load_path)
        self.subjects = self._get_subjects(n_subjects=13)
        self.paradigm = self._get_paradigm()
        if save:
            self.save_set(save_folder, fname)

    def download_raw(self, store_path):
        """Download raw data from dataset repo url and stored it in a folder.

        Parameters
        ----------
        store_path : str, 
            folder path where raw data will be stored, by default None. data will be stored in working path.
        """
        for f in files:
            download_file(f"{self.url}/{f}", store_path)
        zip_files = glob.glob(f"{store_path}/*.zip")
        unzip_files(zip_files, store_path)        
        

    def _get_subjects(self, n_subjects=0):
        """Construct Subjects list

        Parameters
        ----------
        n_subjects : int
            subjects count in DataSet, by default 0

        Returns
        -------
        list of Subject instance
        """
        return [Subject(id='S' + str(s), gender='M', age=0, handedness='')
                for s in range(1, n_subjects + 1)]

    def _get_paradigm(self):
        return ERP(title='ERP_FRIEBURG', stimulation=100,
                   break_duration=150, repetition=68,
                   phrase='Franzy jagt im komplett verwahrlosten Taxi quer durch Freiburg',
                   flashing_mode='SC',
                   speller=[])

    def get_path(self):
        NotImplementedError
