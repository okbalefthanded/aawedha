from aawedha.evaluation.evaluation_utils import aggregate_results
from aawedha.evaluation.evaluation_utils import create_split
from sklearn.model_selection import KFold, StratifiedKFold
from aawedha.evaluation.benchmark import BenchMark
import numpy as np


class SingleSubject(BenchMark):
    """Single Subject Evaluation

    derived from base Evaluation class, takes same attributes and overrides
    generate_split() and run_evaluation().

    Methods
    -------
    _single_subject() for evaluation a single subject data at a time following
    the same folds split generated.

    _fuse_data() when the dataset is beforehand split into train/test sets
     with different subjects for each set, this method concatenates the
     subsets into a single set.
    """

    def generate_split(self, nfolds=30, strategy='Kfold'):
        """Generate cross-validation folds following a cross-validation
        strategy from { Kfold | Stratified } #ShuffleSplit

        Parameters
        ----------
        nfolds : int
            number of cross-validation folds to generate
            default : 30

        strategy : str
            cross-validation strategy
            default : 'Kfold'
        """
        self.settings.nfolds = nfolds
        if self.settings.partition:
            train_phase, val_phase, test_phase, n_trials = self._phases_partition()
        
            self.settings.folds = self.get_folds(nfolds, n_trials, train_phase,
                                    val_phase, test_phase, strategy)
            
    def get_folds(self, nfolds=4, n_trials=0, tr=0, vl=0, ts=0, stg='Kfold'):
        """Generate folds following a KFold cross-validation strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate
            default : 4

        n_trials : int | nd array
            - int : same number of trials across all dataset epochs
            - nd array : each subject has a specific number of trials in
            dataset epochs
            default : 0

        tr : int
            number of training trials to select
            default : 0

        vl : int
            number of validation trials to select
            default : 0

        ts : int
            number of test trials to select
            default : 0

        Returns
        -------
        folds : list of arrays
            each fold has 3 arrays : one for train, one validation, one for
            test, if an independent test set is available in the dataset,
            this will have only 2 array instead of 3
        """
        if isinstance(n_trials, np.ndarray):
            t = [np.arange(n) for n in n_trials]
            folds = [self._get_split(nfolds, n, tr, vl, stg, i) for i, n in enumerate(t)]
        else:
            t = np.arange(n_trials)
            folds = self._get_split(nfolds, t, tr, vl, stg)
        return folds

    def _phases_partition(self):

        n_phase = len(self.settings.partition)
        train_phase = self.settings.partition[0]
        #
        if isinstance(self.dataset.y, list):
            #
            n_trials = np.array([self.dataset.y[i].shape[0]
                                 for i in range(self.n_subjects)])
        else:
            n_trials = self.dataset.y.shape[1]
        #
        if n_phase == 2:
            if hasattr(self.dataset, 'test_epochs'):
                val_phase, test_phase = self.settings.partition[1], 0
            else:
                val_phase, test_phase = 0, self.settings.partition[1]
            # independent test set available
        elif n_phase == 3:
            # generate a set set from the dataset
            val_phase, test_phase = self.settings.partition[1], self.settings.partition[2]
        else:
            # error : wrong settings.partition
            raise AssertionError('Wrong partition scheme', self.settings.partition)
        #
        part = np.round(n_trials / np.sum(self.settings.partition)).astype(int)
        #
        train_phase = train_phase * part
        val_phase = val_phase * part
        test_phase = test_phase * part
        return train_phase, val_phase, test_phase, n_trials
    
    def _eval_operation(self, op):
        """Evaluate a subject on each fold

        Parameters
        ----------
        op : int
            subject id to be selected and evaluated

        Returns
        -------
        subj_results : list of tuple, length = nfolds
            contains subject's performance on each folds
        """
        indie = False
        if hasattr(self.dataset, 'test_epochs'):
            if self._equal_subjects():
                # independent_test = True
                indie = True
            # else:
                # concatenate train & test data
                # test data are different subjects
                # self.n_subjects = self._fuse_data()

        x, y = self._get_data_pair(op)

        subj_results = []
        
        '''
        if isinstance(self.dataset.epochs, list):
            folds_range = range(len(self.settings.folds[0]))
        else:
            folds_range = range(len(self.settings.folds))
        '''
        folds_range = range(self.settings.nfolds)

        for fold in folds_range:
            split = self._split_set(x, y, op, fold, indie)
            split_perf = self._eval_split(split)
            if self.settings.paradigm_metrics:
                # TODO
                paradigm_perf = self._eval_paradigm_metrics(split_perf['probs'], op)
                for m in paradigm_perf:
                    split_perf[m] = paradigm_perf[m]
            subj_results.append(split_perf)
            del split
            # self.learner.reset_weights() # uncessary, the reset_weights is called in _eval_model()

        subj_results = aggregate_results(subj_results)
        return subj_results

    def _fuse_data(self):
        """Concatenate train and test dataset in a single dataset

        Sets dataset.epochs and dataset.y

        Parameters
        ----------
        None

        Returns
        -------
        int : number of subjects in the newly formed dataset by concatenation

        """
        # TODO : when epochs/y/test_epochs/y_test are lists???
        # make lists of lists
        if isinstance(self.dataset.epochs, list):
            # TODO            
            pass
        else:
            self.dataset.epochs = np.vstack((self.dataset.epochs, self.dataset.test_epochs))
            self.dataset.y = np.vstack((self.dataset.y, self.dataset.test_y))
        return self.dataset.epochs.shape[0]  # n_subject

    def _split_set(self, x=None, y=None, subj=0, fold=0, indie=False):
        """Splits Subject data to be evaluated into train/validation/test
        sets following the indices specified in the fold

        Parameters
        ----------
        x : nd array (trials x channels x samples)
            subject EEG data to be split
            default: None

        y : nd array (trials x n_classes)
            subject data labels
            default: None

        subj : int
            subject index in dataset
            default : 0

        fold : int
            fold index in in folds
            default : 0

        indie : bool
            True if independent data set is available, False otherwise
            default : False

        Returns
        -------
        split : dict of nd arrays
            X_train, Y_train, X_Val, Y_Val, X_test, Y_test
            train/validation/test EEG data and labels
        """
        # folds[0][0][0] : inconsistent fold subject trials
        # folds[0][0]    : same trials numbers for all subjects
        split = {}
        
        if self.settings.folds:

            if isinstance(self.dataset.epochs, list):
                folds = self.settings.folds[subj][fold][:]  # subject, fold, phase
            else:
                folds = self.settings.folds[fold][:]
            '''
            if x.ndim == 4:
                trials, kernels, channels, samples = x.shape
            elif x.ndim == 3:
                trials, kernels, samples = x.shape
            '''
            _train, _val, _test = 0, 1, 2
            X_train = x[folds[_train]]
            Y_train = y[folds[_train]]

        else:
            X_train = x
            Y_train = y
        
        X_val, Y_val = None, None
        
        if indie: # independent Test set            
            X_test = self.dataset.test_epochs[subj]
            # _, _, _, channels_format = self._get_fit_configs()
            if X_test.ndim == 3:
                # if channels_format == 'channels_first':
                #   shape = (2, 1, 0)
                shape = (2, 1, 0)
            else:
                shape = (1, 0)
            X_test = X_test.transpose(shape)
            # validation data
            if self.settings.partition and len(self.settings.partition) == 2:
                X_val = x[folds[_val]]
                Y_val = y[folds[_val]]
            Y_test = self.dataset.test_y[subj][:]
        else:
            if len(self.settings.partition) == 2:
                X_test = x[folds[_val]]
                Y_test = y[folds[_val]]
            else:
                X_val = x[folds[_val]]
                Y_val = y[folds[_val]]
                X_test = x[folds[_test]]
                Y_test = y[folds[_test]]

        split = create_split(X_train, X_val, X_test, Y_train, Y_val, Y_test)
        return split

    def _get_data_pair(self, subj=0):
        """Get data pair for a subject from the dataset.
        Transform x, y to model input format.

        Parameters
        ----------
        subj : int
            Subject index in dataset

        Returns
        -------
        x : nd array
            Subject data for evaluation (trials x channels x samples)
        y : array
            class labels
        """
        # prepare data
        x = self.dataset.epochs[subj]
        # _, _, _, channels_format = self._get_fit_configs()
        # shape = (0, 1, 2)
        if x.ndim == 3:
            # if channels_format == 'channels_first':
            #    shape = (2, 1, 0)
            shape = (2, 1, 0)
        else:
            shape = (1, 0)
        x = x.transpose(shape)
        y = self.dataset.y[subj][:]
        return x, y

    def _get_split(self, nfolds, t, tr, vl, stg, subj=None):
        """Generate nfolds following a specified cross-validation strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate

        t : int
            number of total trials

        tr : int
            number of training trials to select

        vl : int
            number of validation trials to select

        stg : str
            cross-validation strategy : K-fold | Stratified k-fold

        Returns
        -------
        folds : list of arrays
            each fold has 3 arrays : one for train, one validation, one for
             test
            if an independent test set is available in the dataset,
            this will have only 2 array instead of 3
        """
        folds = []

        if stg == 'Kfold':
            cv = KFold(n_splits=nfolds, shuffle=True).split(t)
        elif stg == 'Stratified':
            if type(subj) is int:
                y = self.dataset.y[subj]
            else:
                y = self.dataset.y[0]
            # trs = np.arange(0, t)
            cv = StratifiedKFold(n_splits=nfolds).split(t, y)
        
        # for train, test in cv.split(t):
        for train, test in cv:
            if len(self.settings.partition) == 2:
                # no validation split or idependent test set
                folds.append([train, test])
            elif len(self.settings.partition) == 3:
                # generate train/val/test sets from the entire set
                if stg == 'Stratified':
                    if np.sum(np.diff(y) == 0) > np.sum(np.diff(y) == 1):
                        tmp = np.random.choice(train, tr, replace=False)
                        idx = ~np.isin(train, tmp, assume_unique=True)
                        folds.append([tmp, train[idx], test])
                    else:
                        folds.append([train[:tr], train[tr:tr + vl], test])
                else:
                    if (type(tr) is np.ndarray) and (type(vl) is np.ndarray):
                        tr, vl = tr[subj], vl[subj]
                    folds.append([train[:tr], train[tr:tr + vl], test])
                
        return folds

    def _total_operations(self):
        return self.n_subjects

    def _eval_type(self):
        return "Subject"
