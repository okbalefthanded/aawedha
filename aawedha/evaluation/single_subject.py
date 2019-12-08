from aawedha.evaluation.base import Evaluation
from sklearn.model_selection import KFold
import numpy as np


class SingleSubject(Evaluation):
    #
    def generate_split(self, nfolds=30, strategy='Kfold'):
        '''
        '''
        # folds = []
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

        if strategy == 'Kfold':
            self.folds = self._get_folds(nfolds, n_trials, train_phase,
                                         val_phase, test_phase)
        elif strategy == 'Shuffle':
            self.folds = self.get_folds(nfolds, n_trials, train_phase,
                                        val_phase, test_phase,
                                        exclude_subj=False)

    def run_evaluation(self, subject=None):
        '''
        '''
        # generate folds if folds are empty
        if not self.folds:
            self.folds = self.generate_split(nfolds=30)
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

        if subject:
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

        if self.dataset.epochs.ndim == 3:
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
        '''
        '''
        # prepare data
        kernels = 1  #
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
        for fold in range(len(self.folds)):
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
        '''
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
        '''
        '''
        # folds[0][0][0] : inconsistent fold subject trials
        # folds[0][0] : same trials numbers for all subjects
        split = {}
        #
        if isinstance(self.dataset.epochs, list):
            f = self.folds[fold][subj][:]
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

    def _get_folds(self, nfolds=4, n_trials=0, tr=0, vl=0, ts=0):
        '''
        '''
        if isinstance(n_trials, np.ndarray):
            t = [np.arange(n) for n in n_trials]
            sbj = []
            folds = []
            sbj = [self._get_split(nfolds, n, tr, vl) for n in t]
            folds.append(sbj)
        else:
            t = np.arange(n_trials)
            folds = self._get_split(nfolds, t, tr, vl)
        return folds

    def _get_split(self, nfolds, t, tr, vl):
        '''
        '''
        folds = []
        cv = KFold(n_splits=nfolds)
        for train, test in cv.split(t):
            if len(self.partition) == 2:
                # independent test set
                folds.append([train, test])
            elif len(self.partition) == 3:
                # generate test set from the entire set
                folds.append([train[:tr], train[tr:tr + vl], test])
        return folds
