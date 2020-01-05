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
        '''Generate cross-validation folds following a cross-validation
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
        '''

        if self._assert_partiton(excl):
            raise Exception(f'Parition exceeds subjects count, use a different parition')

        # folds = []
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

    def run_evaluation(self, pointer=None, check=False):
        '''Perform evaluation on subsets of subjects

        Parameters
        ----------
        None

        Returns
        -------
        no value, sets results attribute
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.folds = self.generate_split(nfolds=30)
              
        if not pointer and check:
            pointer = CheckPoint(self)

        res_acc, res_auc = [], []
        res_tp, res_fp = [], []

        if not self.model_compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.handlers[0].baseFilename}')
            self.log_experiment()

        if self.current:
            operations = range(self.current, len(self.folds))
        else:
            operations = range(len(self.folds))
        
        # for fold in range(len(self.folds)):
        for fold in operations:

            if self.verbose == 0:
                print(f'Evaluating fold: {fold+1}/{len(self.folds)}...')

            rets = self._cross_subject(fold)
            if isinstance(rets, tuple):
                res_acc.append(rets[0])
                res_auc.append(rets[1])
                res_fp.append(rets[2])
                res_tp.append(rets[3])
            else:
                res_acc.append(rets)

            if self.log:
                msg = f' Fold : {fold+1} ACC: {res_acc[-1]}'
                if len(self.model.metrics) > 1:
                    msg += f' AUC: {res_auc[-1]}'
                self.logger.debug(msg)

            if check:
                pointer.set_checkpoint(fold+1, self.model)

        if isinstance(self.dataset.epochs, np.ndarray) and self.dataset.epochs.ndim == 3:
            #
            self.dataset.recover_dim()

        # Aggregate results
        if res_auc:
            res = np.array([res_acc, res_auc])
            tfpr = np.array([res_fp, res_tp])
        else:
            res = np.array(res_acc)
            tfpr = []

        self.results = self.results_reports(res, tfpr)

    def _cross_subject(self, fold):
        '''Evaluate the subsets of subjects frawn from fold

        Parameters
        ----------
        fold : int
            fold index

        Returns
        -------
        rets : tuple
            folds performance
        '''
        #
        split = self._split_set(fold)
        # normalize data
        X_train, mu, sigma = self.fit_scale(split['X_train'])
        X_val = self.transform_scale(split['X_val'], mu, sigma)
        X_test = self.transform_scale(split['X_test'], mu, sigma)
        Y_train = split['Y_train']
        Y_val = split['Y_val']
        Y_test = split['Y_test']
        #
        cws = self.class_weights(np.argmax(Y_train, axis=1))
        # evaluate model on subj on all folds
        self.model_history, probs = self._eval_model(X_train, Y_train,
                                                     X_val, Y_val, X_test,
                                                     cws)
        # probs = self.model.predict(X_test)
        rets = self.measure_performance(Y_test, probs)

        return rets

    def _split_set(self, fold):
        '''Splits subsets of Subjects data to be evaluated into train/validation/test sets following
        the indices specified in the fold

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
        '''
        split = {}
        kernels = 1
        if isinstance(self.dataset.epochs, list):
            # TODO
            X_train, Y_train = self._cat_lists(fold, 0)
            X_val, Y_val = self._cat_lists(fold, 1)
            X_test, Y_test = self._cat_lists(fold, 2)
            classes = np.unique(Y_train)
            samples, channels, _ = X_train.shape
            X_train = X_train.transpose((2, 1, 0)).reshape(
                (X_train.shape[2], kernels, channels, samples))
            X_val = X_val.transpose((2, 1, 0)).reshape(
                (X_val.shape[2], kernels, channels, samples))
            X_test = X_test.transpose((2, 1, 0)).reshape(
                (X_test.shape[2], kernels, channels, samples))
            Y_train = self.labels_to_categorical(Y_train)
            Y_val = self.labels_to_categorical(Y_val)
            Y_test = self.labels_to_categorical(Y_test)

        else:
            x = self.dataset.epochs
            subjects, samples, channels, trials = x.shape
            y = self.dataset.y
            x = x.transpose((0, 3, 2, 1))
            #
            classes = np.unique(y)
            y = self.labels_to_categorical(y)
            # n_subjects per train/val
            tr, val = self.partition[0], self.partition[1]
            if len(self.partition) == 3:
                ts = self.partition[2]

            X_train = x[self.folds[fold][0], :, :, :].reshape(
                (tr * trials, kernels, channels, samples))
            X_val = x[self.folds[fold][1], :, :, :].reshape(
                (val * trials, kernels, channels, samples))

            ctg_dim = y.shape[2]
            Y_train = y[self.folds[fold][0], :].reshape((tr * trials, ctg_dim))
            Y_val = y[self.folds[fold][1], :].reshape((val * trials, ctg_dim))

            if hasattr(self.dataset, 'test_epochs'):
                trs = self.dataset.test_epochs.shape[3]
                X_test = self.dataset.test_epochs.transpose((0, 3, 2, 1))
                X_test = X_test.reshape(
                    (trs * self.n_subjects, kernels, channels, samples))
                # Y_test = tf_utils.to_categorical(self.dataset.test_y)
                Y_test = self.labels_to_categorical(
                    self.dataset.test_y.reshape((self.n_subjects * trs)))
            else:
                X_test = x[self.folds[fold][2], :, :, :].reshape(
                    (ts * trials, kernels, channels, samples))
                Y_test = y[self.folds[fold][2], :].reshape(
                    (ts * trials, ctg_dim))

        split['X_train'] = X_train
        split['X_val'] = X_val
        split['X_test'] = X_test
        split['Y_train'] = Y_train
        split['Y_val'] = Y_val
        split['Y_test'] = Y_test
        split['classes'] = classes
        return split

    def _cat_lists(self, fold=0, phase=0):
        '''Concatenate lists into a single Tensor

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
        '''
        if phase == 2:  # test phase
            if hasattr(self.dataset, 'test_epochs'):
                X = np.concatenate([self.dataset.test_epochs[idx]
                                   for idx in range(self.n_subjects)], axis=-1)
                Y = np.concatenate([self.dataset.test_y[idx]
                                   for idx in range(self.n_subjects)], axis=-1)
                return X, Y
        X = np.concatenate([self.dataset.epochs[idx]
                            for idx in self.folds[fold][phase]], axis=-1)
        Y = np.concatenate([self.dataset.y[idx]
                            for idx in self.folds[fold][phase]], axis=-1)
        return X, Y
