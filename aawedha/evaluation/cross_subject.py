from aawedha.evaluation.base import Evaluation
from aawedha.evaluation.checkpoint import CheckPoint
import numpy as np


class CrossSubject(Evaluation):
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

        Returns
        -------
        no value, sets folds attribute with a list of arrays
        """

        if self._assert_partition(excl):
            raise Exception(
                f'Parition exceeds subjects count, use a different parition')

        n_phase = len(self.partition)
        train_phase, val_phase = self.partition[0], self.partition[1]

        if n_phase == 2:
            # independent test set available
            test_phase = 0
        elif n_phase == 3:
            # generate a set set from the dataset
            test_phase = self.partition[2]
        else:
            # error : wrong partition
            raise AssertionError('Wrong partition scheme', self.partition)

        self.folds = self.get_folds(
            nfolds, self.n_subjects, train_phase, val_phase,
            test_phase, exclude_subj=excl)

    def run_evaluation(self, folds=None, pointer=None, check=False):
        """Perform evaluation on subsets of subjects

        Parameters
        ----------
        pointer : CheckPoint instance
            save state of evaluation

        check : bool
            if True, sets evaluation checkpoint for future operation resume,
            False otherwise

        Returns
        -------
        no value, sets results attribute
        """
        # generate folds if folds are empty
        if not self.folds:
            self.generate_split(nfolds=30)

        if not pointer and check:
            pointer = CheckPoint(self)

        res = []
        if not self.model_compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.handlers[0].baseFilename}')
            self.log_experiment()

        operations = self.get_operations(folds)

        for fold in operations:
            #
            if self.verbose == 0:
                print(f'Evaluating fold: {fold+1}/{len(self.folds)}...')

            rets = self._cross_subject(fold)
            if self.log:
                msg = f" Subj : {fold+1} ACC: {rets['accuracy']}"
                # if len(self.model.metrics) > 1:
                if len(self.model_config['compile']['metrics']) > 1:
                    msg += f" AUC: {rets['auc']}"
                self.logger.debug(msg)
                self.logger.debug(
                    f' Training stopped at epoch: {self.model_history.epoch[-1]}')

            res.append(rets)

            if check:
                pointer.set_checkpoint(fold+1, self.model)
        #
        if (isinstance(self.dataset.epochs, np.ndarray) and
                self.dataset.epochs.ndim == 3):
            #
            self.dataset.recover_dim()

        if len(self.folds) == len(self.predictions):
            # self.results = self.results_reports(res, tfpr)
            self.results = self.results_reports(res)

    def get_folds(self, nfolds, population, tr, vl, ts, exclude_subj=True):
        """Generate train/validation/tes folds following Shuffle split strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate
        population : int
            number of total trials/subjects available
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
        #
        return folds

    def get_operations(self, folds=None):
        """get an iterable object for evaluation, it can be
        all folds or a defined subset of folds.
        In case of long evaluation, the iterable starts from the current
        index

        Parameters
        ----------
        folds : list | int, optional
            defined list of folds or a just a single one, by default None

        Returns
        -------
        range | list
            selection of folds to evaluate, from all folds available to a
            defined subset
        """
        if self.current and not folds:
            operations = range(self.current, len(self.folds))
        elif type(folds) is list:
            operations = folds
        elif type(folds) is int:
            operations = [folds]
        else:
            operations = range(len(self.folds))

        return operations

    def _cross_subject(self, fold):
        """Evaluate the subsets of subjects drawn from fold

        Parameters
        ----------
        fold : int
            fold index

        Returns
        -------
        rets : tuple
            folds performance
        """
        split = self._split_set(fold)
        rets = self._eval_split(split)
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
        split = {}

        X_train, Y_train = self._cat_lists(fold, 0)
        X_test, Y_test = self._cat_lists(fold, 2)
        X_train = X_train.transpose((2, 1, 0))
        X_test = X_test.transpose((2, 1, 0))

        if self._has_val():
            X_val, Y_val = self._cat_lists(fold, 1)
            X_val = X_val.transpose((2, 1, 0))
        else:
            X_val, Y_val = None, None

        if Y_train.min() != 0:
            Y_train -= 1
            Y_test -= 1
            if Y_val is not None:
                Y_val -= 1

        split['X_train'] = X_train
        split['X_val'] = X_val
        split['X_test'] = X_test
        split['Y_train'] = Y_train
        split['Y_val'] = Y_val
        split['Y_test'] = Y_test
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
                                    for idx in self.folds[fold][phase]], axis=-1)
                Y = np.concatenate([self.dataset.test_y[idx]
                                    for idx in self.folds[fold][phase]], axis=-1)
                return X, Y
        X = np.concatenate([self.dataset.epochs[idx]
                            for idx in self.folds[fold][phase]], axis=-1)
        Y = np.concatenate([self.dataset.y[idx]
                            for idx in self.folds[fold][phase]], axis=-1)
        return X, Y
