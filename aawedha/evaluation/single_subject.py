from aawedha.evaluation.base import Evaluation
from aawedha.evaluation.checkpoint import CheckPoint
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


class SingleSubject(Evaluation):
    '''Single Subject Evaluation

    derived from base Evaluation class, takes same attributes and overrides
    generate_split() and run_evaluation().

    Methods
    -------
    _single_subject() for evaluation a single subject data at a time following the
        same folds split generated.

    _fuse_data() when the dataset is beforehand split into train/test sets with different subjects
          for each set, this method concatenates the subsets into a single set.
    '''

    def generate_split(self, nfolds=30, strategy='Kfold'):
        '''Generate cross-validation folds following a cross-validation
        strategy from { Kfold | ShuffleSplit }

        Parameters
        ----------
        nfolds : int
            number of cross-validation folds to generate
            default : 30

        strategy : str
            cross-validation strategy
            default : 'Kfold'

        Returns
        -------
        no value, sets folds attribute with a list of arrays
        '''
        n_phase = len(self.partition)
        train_phase, val_phase = self.partition[0], self.partition[1]
        #
        if isinstance(self.dataset.y, list):
            #
            n_trials = np.array([self.dataset.y[i].shape[0]
                                 for i in range(self.n_subjects)])
        else:
            n_trials = self.dataset.y.shape[1]
        #
        if n_phase == 2:
            # independent test set available
            test_phase = 0
        elif n_phase == 3:
            # generate a set set from the dataset
            test_phase = self.partition[2]
        else:
            # error : wrong partition
            raise AssertionError('Wrong partition scheme', self.partition)
        #
        part = np.round(n_trials / np.sum(self.partition)).astype(int)
        #
        train_phase = train_phase * part
        val_phase = val_phase * part
        test_phase = test_phase * part

        
        if strategy == 'Shuffle':
            self.folds = self.get_folds(nfolds, n_trials, train_phase,
                                        val_phase, test_phase,
                                        exclude_subj=False)
        else:
            self.folds = self._get_folds(nfolds, n_trials, train_phase,
                                         val_phase, test_phase, strategy)

    def run_evaluation(self, subject=None, pointer=None, check=False):
        '''Perform evaluation on each subject

        Parameters
        ----------
        subject : int | list
            - specific subject id, performs a single evaluation.
            - list of subjects from the set of subjects available in dataset
            default : None, evaluate each subject

        checkpoint : bool
            if True, sets evaluation checkpoint for future operation resume, False otherwise

        Returns
        -------
        no value, sets results attribute
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.generate_split(nfolds=30)
        
        if not pointer and check:
            pointer = CheckPoint(self)
        #
        res_acc = []
        res_auc, res_tp, res_fp = [], [], []

        independent_test = False

        if hasattr(self.dataset, 'test_epochs'):
            if self._equale_subjects():
                independent_test = True
            else:
                # concatenate train & test data
                # test data are different subjects
                n_subj = self._fuse_data()
                self.n_subjects = n_subj
        #
        if self.current:
            operations = range(self.current, self.n_subjects)
        elif subject:
            operations = [subject]
        else:
            operations = range(self.n_subjects)
        #
        if not self.model_compiled:
            self._compile_model()

        if self.log:
            print(f'Logging to file : {self.logger.handlers[0].baseFilename}')
            self.log_experiment()

        for subj in operations:
            #
            if self.verbose == 0:
                print(f'Evaluating Subject: {subj+1}/{len(operations)}...')

            rets = self._single_subject(subj, independent_test)
            if isinstance(rets[0], tuple):
                res_acc.append([elm[0] for elm in rets])
                res_auc.append([elm[1] for elm in rets])
                res_fp.append([elm[2] for elm in rets])
                res_tp.append([elm[3] for elm in rets])
            else:
                res_acc.append(rets)

            if self.log:
                msg = f' Subj : {subj+1} ACC: {res_acc[-1]}'
                if len(self.model.metrics) > 1:
                    msg += f' AUC: {res_auc[-1]}'
                self.logger.debug(msg)
                self.logger.debug(f' Training stopped at epoch: {self.model_history.epoch[-1]}')

            if check:
                pointer.set_checkpoint(subj+1, self.model)

        if not isinstance(self.dataset.epochs, list) and self.dataset.epochs.ndim == 3:
            self.dataset.recover_dim()

        # Aggregate results
        tfpr = {}
        if res_auc:
            res = {}
            res['acc'] = res_acc
            res['auc'] = res_auc
            tfpr['fp'] = res_fp
            tfpr['tp'] = res_tp
        else:
            res = np.array(res_acc)
        #
        self.results = self.results_reports(res, tfpr)       

    def _single_subject(self, subj, indie=False):
        '''Evaluate a subject on each fold

        Parameters
        ----------
        subj : int
            subject id to be selected and evaluated

        indie : bool
            True if independent set available in dataset, so no need to use the test fold.
            default : False

        Returns
        -------
        rets : list of tuple, length = nfolds
            contains subject's performance on each folds
        '''
        # prepare data
        kernels = 1  #
        if isinstance(self.dataset.epochs, list):
            # TODO
            x = self.dataset.epochs[subj]
            samples, channels, trials = x.shape
            x = x.transpose((2, 1, 0)).reshape((trials, kernels, channels, samples))
        else:
            if self.dataset.epochs.ndim == 4:
                x = self.dataset.epochs[subj][:, :, :]
                samples, channels, trials = x.shape
                x = x.transpose((2, 1, 0)).reshape((trials, kernels, channels, samples))
            elif self.dataset.epochs.ndim == 3:
                x = self.dataset.epochs[subj][:, :]
                samples, trials = x.shape
                x = x.transpose((1, 0)).reshape((trials, kernels, samples))

        # x = x.reshape((trials, kernels, channels, samples))
        y = self.dataset.y[subj][:]
        y = self.labels_to_categorical(y)
        rets = []
        # get in the fold!!!
        if isinstance(self.dataset.epochs, list):
            folds_range = range(len(self.folds[0]))
        else:
            folds_range = range(len(self.folds))

        # for fold in range(len(self.folds)):
        for fold in folds_range:
            #
            split = self._split_set(x, y, subj, fold, indie)
            # normalize data
            X_train, mu, sigma = self.fit_scale(split['X_train'])
            X_val = self.transform_scale(split['X_val'], mu, sigma)
            X_test = self.transform_scale(split['X_test'], mu, sigma)
            '''
            X_train = split['X_train']
            X_val = split['X_val']
            X_test = split['X_test']
            '''
            #
            Y_train = split['Y_train']
            Y_test = split['Y_test']
            Y_val = split['Y_val']
            #
            class_weights = self.class_weights(np.argmax(Y_train, axis=1))
            # evaluate model on subj on all folds
            self.model_history, probs = self._eval_model(X_train, Y_train,
                                                         X_val, Y_val, X_test,
                                                         class_weights)

            # probs = self.model.predict(X_test)
            rets.append(self.measure_performance(Y_test, probs))

        return rets

    def _fuse_data(self):
        '''Concatenate train and test dataset in a single dataset

        Sets dataset.epochs and dataset.y

        Parameters
        ----------
        None

        Returns
        -------
        int : number of subjects in the newly formed dataset by concatenation

        '''
        # TODO : when epochs/y/test_epochs/y_test are lists???
        # make lists of lists
        if isinstance(self.dataset.epochs, list):
            # TODO
            pass
        else:
            self.dataset.epochs = np.vstack(
                (self.dataset.epochs, self.dataset.test_epochs))
            self.dataset.y = np.vstack((self.dataset.y, self.dataset.test_y))
        return self.dataset.epochs.shape[0]  # n_subject

    def _split_set(self, x=None, y=None, subj=0, fold=0, indie=False):
        '''Splits Subject data to be evaluated into train/validation/test sets following
        the indices specified in the fold

        Parameters
        ----------
        x : nd array (trials x kernels x channels x samples)
            subject EEG data to be split
            default: None

        y : nd array (n_examples x n_classes)
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
            defaul : False

        Returns
        -------
        split : dict of nd arrays
            X_train, Y_train, X_Val, Y_Val, X_test, Y_test
            train/validation/test EEG data and labels
        '''
        # folds[0][0][0] : inconsistent fold subject trials
        # folds[0][0] : same trials numbers for all subjects
        split = {}
        #
        if isinstance(self.dataset.epochs, list):
            # f = self.folds[fold][subj][:]
            f = self.folds[subj][fold][:]  # subject, fold, phase
        else:
            f = self.folds[fold][:]

        if x.ndim == 4:
            trials, kernels, channels, samples = x.shape
        elif x.ndim == 3:
            trials, kernels, samples = x.shape

        X_train = x[f[0]]
        X_val = x[f[1]]
        Y_train = y[f[0]]
        Y_val = y[f[1]]
        if indie:
            if isinstance(self.dataset.test_epochs, list):
                # TODO
                trs = self.dataset.test_epochs[0].shape[2]
                X_test = self.dataset.test_epochs[subj].transpose((2, 1, 0))
                X_test = X_test.reshape((trs, kernels, channels, samples))
            else:
                if self.dataset.test_epochs.ndim == 3:
                    sbj, s, t = self.dataset.test_epochs.shape
                    # self.dataset.test_epochs = self.dataset.test_epochs.reshape((sbj, s, 1, t))
                    X_test = self.dataset.test_epochs[subj].transpose((1, 0))
                    X_test = X_test.reshape((t, kernels, samples))
                elif self.dataset.test_epochs.ndim == 4:
                    trs = self.dataset.test_epochs[0].shape[2]
                    X_test = self.dataset.test_epochs[subj][:, :, :].transpose((2, 1, 0))
                    X_test = X_test.reshape((trs, kernels, channels, samples))

            Y_test = self.labels_to_categorical(self.dataset.test_y[subj][:])
        else:
            X_test = x[f[2]]
            Y_test = y[f[2]]

        split['X_train'] = X_train
        split['Y_train'] = Y_train
        split['X_test'] = X_test
        split['Y_test'] = Y_test
        split['X_val'] = X_val
        split['Y_val'] = Y_val
        return split

    def _get_folds(self, nfolds=4, n_trials=0, tr=0, vl=0, ts=0, stg='Kfold'):
        '''Generate folds following a KFold cross-validation strategy

        Parameters
        ----------
        nfolds : int
            number of folds to generate
            default : 4

        n_trials : int | nd array
            - int : same number of trials across all dataset epochs
            - nd array : each subject has a specific number of trials in dataset epochs
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
            each fold has 3 arrays : one for train, one validation, one for test
            if an independent test set is available in the dataset, this will have
            only 2 array instead of 3
        '''
        if isinstance(n_trials, np.ndarray):
            t = [np.arange(n) for n in n_trials]          
            # folds = []
            # sbj = [self._get_split(nfolds, n, tr, vl) for n in t]
            # folds.append(sbj)
            folds = [self._get_split(nfolds, n, tr, vl, stg) for n in t]
        else:
            t = np.arange(n_trials)
            folds = self._get_split(nfolds, t, tr, vl,stg)
        return folds

    def _get_split(self, nfolds, t, tr, vl, stg):
        '''Generate nfolds following a specified cross-validation strategy

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
            each fold has 3 arrays : one for train, one validation, one for test
            if an independent test set is available in the dataset, this will have
            only 2 array instead of 3
        '''
        folds = []
        if stg == 'Kfold':        
            cv = KFold(n_splits=nfolds).split(t)
        elif stg  == 'Stratified':
            y = self.dataset.y[0]
            # trs = np.arange(0, t)            
            cv = StratifiedKFold(n_splits=nfolds).split(t,y)
        # for train, test in cv.split(t):
        for train, test in cv:
            if len(self.partition) == 2:
                # independent test set
                folds.append([train, test])
            elif len(self.partition) == 3:
                # generate test set from the entire set
                folds.append([train[:tr], train[tr:tr + vl], test])
        return folds
