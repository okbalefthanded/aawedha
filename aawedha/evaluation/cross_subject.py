from aawedha.evaluation.evaluation_utils import create_split
from aawedha.evaluation.benchmark import BenchMark
import numpy as np


class CrossSubject(BenchMark):
    '''Cross Subject Evaluation

    derived from base Evaluation class, takes same attributes and overrides
    generate_split() and run_evaluation().

    Methods
    -------
    _cross_subject()
        for evaluation a subset of subjects data at a time
        following the same folds split generated.

    _split_set()
        splits dataset into 3 (or 2 in case of independent test set)
        distinct subset for train/validation/test.

    _cat_lists()
        some datasets has different number of trials for each subject,   
        so the dataset is a list of ndarrays instead of a single Tensor,
        this method concatenates selected subject in a single Tensor at
        evaluation.
    '''
    def generate_split(self, nfolds=30, excl=True):
        """Generate cross-validation folds following a cross-validation
        strategy from ShuffleSplit

        Parameters
        ----------
        nfolds : int
            number of cross-validation folds to generate
            default : 30

        excl : bool
            True, exclude target subject from evaluation to insure a complete
            cross-subject evaluation, False otherwise
            default : True
        """
        if self._assert_partition(excl):
            raise Exception(
                f'Parition exceeds subjects count, use a different parition')

        train_phase, val_phase, test_phase = self._phases_partiton()
        self.settings.nfolds = nfolds
        self.settings.folds  = self.get_folds(nfolds, train_phase, val_phase, 
                                             test_phase, exclude_subj=excl)
        return self

    def get_folds(self, nfolds, tr, vl, ts, exclude_subj=True):
        """Generate train/validation/tes folds following Shuffle split strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate

        tr : int
            number of trials/subjects to include in train folds
        
        vl : int
            number of trials/subjects to include in validation folds
        
        ts : int
            number of trials/subjects to include in test folds

        exclude_subj : bool (only in CrossSubject evaluation)
            if True the target subject data will be excluded from train fold

        Returns
        -------
        folds : list
        """
        folds = []

        # list : nfolds : [nsubjects_train] [nsubjects_val][nsubjects_test]
        for subj in range(self.n_subjects):
            selection = np.arange(0, self.n_subjects)
            if exclude_subj:
                # fully cross-subject, no subject train data in fold
                selection = np.delete(selection, subj)
            for fold in range(nfolds):
                np.random.shuffle(selection)
                folds.append([np.array(selection[:tr]),
                              np.array(selection[tr:tr + vl]),
                              np.array([subj])
                              ])
        return folds

    def _phases_partiton(self):
        n_phase = len(self.settings.partition)
        train_phase, val_phase = self.settings.partition[0], self.settings.partition[1]

        if n_phase == 2:
            # independent test set available
            test_phase = 0
        elif n_phase == 3:
            # generate a set set from the dataset
            test_phase = self.settings.partition[2]
        else:
            # error : wrong settings.partition
            raise AssertionError('Wrong settings.partition scheme', self.settings.partition)
        return train_phase, val_phase, test_phase
    
    def _eval_operation(self, op):
        """Evaluate the subsets of subjects drawn from fold

        Parameters
        ----------
        fold (op) : int
            fold index

        Returns
        -------
        rets : tuple
            folds performance
        """
        split = self._split_set(op)
        rets  = self._eval_split(split)
        del split 
        # self.learner.reset_weights()
        return rets

    def _split_set(self, fold):
        """Splits subsets of Subjects data to be evaluated into
        train/validation/test sets following the indices specified in the fold

        Parameters
        ----------
        fold : int
            fold index

        Returns
        -------
        split : dict of nd arrays
            X_train, Y_train, X_Val, Y_Val, X_test, Y_test
            train/validation/test EEG data and labels
            classes : array of values used to denote class labels
        """
        '''
        shape = (1, 0, 2)
        _, _, _, channels_format = self._get_fit_configs()
        if channels_format == 'channels_first':
            shape = (2, 1, 0)
        '''
        _train, _val, _test = 0, 1, 2
        shape = (2, 1, 0)
        X_train, Y_train = self._cat_lists(fold, _train)
        X_test, Y_test = self._cat_lists(fold, _test)
        X_train = X_train.transpose(shape)
        X_test = X_test.transpose(shape)

        X_val, Y_val = None, None
        if self._has_val():
            X_val, Y_val = self._cat_lists(fold, _val)
            X_val = X_val.transpose(shape)            

        split = create_split(X_train, X_val, X_test, Y_train, Y_val, Y_test)
        return split

    def _cat_lists(self, fold=0, phase=0):
        """Concatenate lists into a single Tensor

        Parameters
        ----------
        fold : int
            fold index

        phase : int
            evaluation phase, 0 for test, 1 for validation
            2 for test

        Returns
        -------
        X : ndarray (subjects*trials, samples, channels)
            EEG data concatenated
        Y : ndarray (subject*n_examples, classes)
            class labels
        """
        if phase == 2:  # test phase
            if hasattr(self.dataset, 'test_epochs'):
                X = np.concatenate([self.dataset.test_epochs[idx] 
                                    for idx in self.settings.folds[fold][phase]], axis=-1)
                Y = np.concatenate([self.dataset.test_y[idx] 
                                    for idx in self.settings.folds[fold][phase]], axis=-1)
                return X, Y
        X = np.concatenate([self.dataset.epochs[idx] 
                            for idx in self.settings.folds[fold][phase]], axis=-1)
        Y = np.concatenate([self.dataset.y[idx] 
                            for idx in self.settings.folds[fold][phase]], axis=-1)
        return X, Y

    def _total_operations(self):
        return len(self.settings.folds)

    def _eval_type(self):
        return "Fold"
